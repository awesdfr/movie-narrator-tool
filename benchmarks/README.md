# Benchmark Manifest

Run the evaluator after a project has finished matching:

```powershell
python backend/cli/benchmark_cli.py --project <project_id> --manifest benchmarks/sample_manifest.json
```

Manifest rules:
- `label: "movie"` requires `movie_start` and `movie_end` ground truth.
- `label: "non_movie"` means the segment should not be matched to a movie clip.
- A segment is counted as correct when `IoU >= 0.8` or both start/end errors are `<= 1.5s`.
- The report also measures false match rate and how often incorrect results were surfaced as review-required.

Recommended scenario coverage:
- crop_or_resize
- subtitle_occlusion
- weak_or_no_audio
- jump_cut_dialogue
- repeated_shot
- non_movie
- short_segment
- long_segment
