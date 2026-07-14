//! Held-out value-network validation metrics.

use std::{
    fs::File,
    io::{BufReader, Error, ErrorKind, Read},
    mem::size_of,
    path::Path,
};

use crate::shogi::PackedSfenValue;

use super::loader::WrmTargetParams;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrmOutputParams {
    pub in_scaling: f32,
    pub out_scaling: f32,
    pub offset: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ValidationParams {
    pub blend: f32,
    pub eval_scale: f32,
    pub wrm_target: Option<WrmTargetParams>,
    pub wrm_output: Option<WrmOutputParams>,
    pub score_drop_abs: Option<u16>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AccuracyReport {
    pub compared: usize,
    pub sign_matches: usize,
    pub drawn_games: usize,
    pub filtered_by_score_cap: usize,
    pub non_finite: usize,
    pub loss_sampled: usize,
    pub test_loss: Option<f32>,
}

impl AccuracyReport {
    pub fn accuracy(&self) -> f32 {
        if self.compared == 0 { f32::NAN } else { self.sign_matches as f32 / self.compared as f32 }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn win_rate(score: f32, scaling: f32, offset: f32) -> f32 {
    let p = sigmoid((score - offset) / scaling);
    let pm = sigmoid((-score - offset) / scaling);
    0.5 * (1.0 + p - pm)
}

/// Computes decisive-game sign accuracy and the mean held-out training loss.
/// Draws are excluded from accuracy but retained in the loss subset.
/// Non-finite network outputs are counted as accuracy mismatches (a NaN output
/// otherwise compares equal to a lost game and inflates accuracy) and excluded
/// from the loss mean.
/// The loss is averaged over surviving positions (finite output, not dropped by
/// `score_drop_abs`); the training log instead divides by the full batch size, so
/// under `--score-drop-abs` the two denominators differ.
pub fn compute_test_metrics(
    model_outputs: &[f32],
    teacher_scores: &[i16],
    teacher_results: &[i8],
    params: ValidationParams,
) -> AccuracyReport {
    assert_eq!(model_outputs.len(), teacher_scores.len(), "model output and score length mismatch");
    assert_eq!(model_outputs.len(), teacher_results.len(), "model output and result length mismatch");

    let mut report = AccuracyReport::default();
    let mut loss_sum = 0.0f64;

    for ((&output, &score), &result) in model_outputs.iter().zip(teacher_scores).zip(teacher_results) {
        if params.score_drop_abs.is_some_and(|cap| score.unsigned_abs() >= cap) {
            report.filtered_by_score_cap += 1;
            continue;
        }

        let output_finite = output.is_finite();
        if !output_finite {
            report.non_finite += 1;
        }

        if result == 0 {
            report.drawn_games += 1;
        } else {
            report.compared += 1;
            if output_finite && (output >= 0.0) == (result > 0) {
                report.sign_matches += 1;
            }
        }

        if !output_finite {
            continue;
        }

        let result_norm = match result.signum() {
            1 => 1.0,
            -1 => 0.0,
            _ => 0.5,
        };
        let score_norm = params.wrm_target.map_or_else(
            || sigmoid(f32::from(score) / params.eval_scale),
            |wrm| win_rate(f32::from(score), wrm.scaling, wrm.offset),
        );
        let target = params.blend * result_norm + (1.0 - params.blend) * score_norm;
        let prediction = params
            .wrm_output
            .map_or_else(|| sigmoid(output), |wrm| win_rate(output * wrm.out_scaling, wrm.in_scaling, wrm.offset));
        let diff = f64::from(prediction - target);
        loss_sum += diff * diff;
        report.loss_sampled += 1;
    }

    if report.loss_sampled > 0 {
        report.test_loss = Some((loss_sum / report.loss_sampled as f64) as f32);
    }
    report
}

/// Reads at most `limit` PSV records in file order. A zero limit reads all records.
pub fn read_psv_positions(path: &Path, limit: usize) -> std::io::Result<Vec<PackedSfenValue>> {
    let record_size = size_of::<PackedSfenValue>();
    let file_size = std::fs::metadata(path)?.len() as usize;
    if !file_size.is_multiple_of(record_size) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("{}: size {file_size} is not a multiple of {record_size}", path.display()),
        ));
    }

    let available = file_size / record_size;
    let count = if limit == 0 { available } else { available.min(limit) };
    let mut reader = BufReader::new(File::open(path)?);
    let mut positions = Vec::with_capacity(count);
    for _ in 0..count {
        let mut position = PackedSfenValue::default();
        reader.read_exact(position.as_bytes_mut())?;
        positions.push(position);
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_params() -> ValidationParams {
        ValidationParams { blend: 1.0, eval_scale: 400.0, wrm_target: None, wrm_output: None, score_drop_abs: None }
    }

    #[test]
    fn decisive_sign_accuracy_excludes_draws() {
        let report =
            compute_test_metrics(&[1.0, -1.0, 0.0, -0.5], &[100, -100, 0, 50], &[1, -1, 0, 1], standard_params());
        assert_eq!(report.compared, 3);
        assert_eq!(report.sign_matches, 2);
        assert_eq!(report.drawn_games, 1);
        assert_eq!(report.loss_sampled, 4);
        assert!((report.accuracy() - 2.0 / 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn zero_output_predicts_win() {
        let report = compute_test_metrics(&[0.0, 0.0], &[0, 0], &[1, -1], standard_params());
        assert_eq!(report.sign_matches, 1);
        assert_eq!(report.compared, 2);
    }

    #[test]
    fn score_cap_excludes_positions_from_both_metrics() {
        let mut params = standard_params();
        params.score_drop_abs = Some(32_000);
        let report = compute_test_metrics(&[1.0, -1.0, 1.0], &[32_000, -32_000, 100], &[1, -1, 1], params);
        assert_eq!(report.filtered_by_score_cap, 2);
        assert_eq!(report.compared, 1);
        assert_eq!(report.loss_sampled, 1);
    }

    #[test]
    fn loss_uses_wdl_blend() {
        let mut params = standard_params();
        params.blend = 0.0;
        let report = compute_test_metrics(&[0.25, -0.25], &[100, -100], &[-1, 1], params);
        assert!(report.test_loss.unwrap() < 1.0e-12);
    }

    #[test]
    fn loss_uses_wrm_target_and_output() {
        let wrm = WrmTargetParams { scaling: 380.0, offset: 270.0 };
        let params = ValidationParams {
            blend: 0.0,
            eval_scale: 600.0,
            wrm_target: Some(wrm),
            wrm_output: Some(WrmOutputParams { in_scaling: 380.0, out_scaling: 600.0, offset: 270.0 }),
            score_drop_abs: None,
        };
        let report = compute_test_metrics(&[0.5, -0.5], &[300, -300], &[1, -1], params);
        assert!(report.test_loss.unwrap() < 1.0e-12);
    }

    #[test]
    fn empty_input_has_no_loss_and_nan_accuracy() {
        let report = compute_test_metrics(&[], &[], &[], standard_params());
        assert_eq!(report.test_loss, None);
        assert!(report.accuracy().is_nan());
    }

    #[test]
    fn nan_output_does_not_masquerade_as_loss_sign_match() {
        // A NaN output on a lost game: (NaN >= 0.0) == false and (result > 0) == false,
        // so without an explicit finite guard the two compare equal and count as a match.
        let report = compute_test_metrics(&[f32::NAN], &[-100], &[-1], standard_params());
        assert_eq!(report.compared, 1);
        assert_eq!(report.sign_matches, 0);
        assert_eq!(report.non_finite, 1);
        assert_eq!(report.loss_sampled, 0);
        assert_eq!(report.test_loss, None);
    }

    #[test]
    fn non_finite_outputs_are_excluded_from_loss_and_counted_as_mismatch() {
        let report =
            compute_test_metrics(&[f32::NAN, f32::INFINITY, 1.0], &[100, -100, 50], &[1, -1, 1], standard_params());
        assert_eq!(report.compared, 3);
        assert_eq!(report.sign_matches, 1);
        assert_eq!(report.non_finite, 2);
        assert_eq!(report.loss_sampled, 1);
        assert!(report.test_loss.unwrap().is_finite());
    }
}
