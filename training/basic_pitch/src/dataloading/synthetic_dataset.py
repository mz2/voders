from pathlib import Path

from .base_dataloader import PianoRollAudioDataset


class SyntheticDataset(PianoRollAudioDataset):
    def __str__(self):
        return "Synthetic"

    def __init__(
        self,
        path="syntheticdataset",
        groups=None,
        sequence_length=None,
        seed=42,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            path,
            groups if groups is not None else ["train"],
            sequence_length,
            seed,
            device,
            **kwargs,
        )

    @classmethod
    def available_groups(cls):
        return ["train"]

    def files(self, group):
        if group not in self.available_groups():
            raise ValueError(f"Group {group} does not exist")

        root = Path(self.path)
        if not root.is_dir():
            return []

        result = []
        seen = set()

        # Original challenge layout: one directory per sample containing the
        # fixed audio.wav / score.tsv names.
        for song_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            audio_path = song_dir / "audio.wav"
            tsv_path = song_dir / "score.tsv"
            if audio_path.exists() and tsv_path.exists():
                result.append((audio_path, tsv_path))
                seen.add(audio_path.resolve())

        # voders layout: sharded files whose paired WAV/TSV share a stem.
        for audio_path in sorted(root.rglob("*.wav")):
            if audio_path.resolve() in seen:
                continue
            tsv_path = audio_path.with_suffix(".tsv")
            if tsv_path.exists():
                result.append((audio_path, tsv_path))
        return result

    def __getitem__(self, index):
        result = super().__getitem__(index)
        data_index, _ = self.chunk_indices[index]
        result["song_id"] = Path(self.data[data_index]["path"]).stem
        return result


if __name__ == "__main__":
    import soundfile as sf

    from ..constants import SAMPLE_RATE

    DATA_ROOT = "syntheticdataset"
    GROUPS = ["train"]
    SONG_INDEX = 0
    OUT_WAV = "synthetic_song_example.wav"

    ds = SyntheticDataset(path=DATA_ROOT, groups=GROUPS, sequence_length=None)
    if len(ds) == 0:
        print(f"No samples found in {DATA_ROOT}")
        raise SystemExit(0)

    item = ds[SONG_INDEX]
    audio = item["audio"].squeeze(0).cpu().numpy()

    sf.write(OUT_WAV, audio, SAMPLE_RATE)
    print(f"Wrote {OUT_WAV}")
