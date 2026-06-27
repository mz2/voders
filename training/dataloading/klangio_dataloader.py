from pathlib import Path

from .base_dataloader import PianoRollAudioDataset

# Klangio validation/test audio ships in the MML26-singing-synthesis repo, vendored
# here as a git submodule. Anchor to the repo root so the default resolves regardless
# of the process working directory (validation has no --val-path override).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_KLANGIO_PATH = _REPO_ROOT / "external" / "MML26-singing-synthesis" / "klangiodataset"


class KlangioDataset(PianoRollAudioDataset):
    def __str__(self):
        return "Klangio"

    def __init__(
        self,
        path=str(_DEFAULT_KLANGIO_PATH),
        groups=None,
        sequence_length=None,
        seed=42,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            path,
            groups if groups is not None else ["test"],
            sequence_length,
            seed,
            device,
            **kwargs,
        )

    @classmethod
    def available_groups(cls):
        return ["test"]

    def files(self, group):
        if group not in self.available_groups():
            raise ValueError(f"Group {group} does not exist")

        result = []
        for audio_path in sorted(Path(self.path).glob("*.wav")):
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

    DATA_ROOT = "klangiodataset"
    GROUPS = ["test"]
    SONG_INDEX = 0
    OUT_WAV = "klangio_song_example.wav"

    ds = KlangioDataset(path=DATA_ROOT, groups=GROUPS, sequence_length=None)

    item = ds[SONG_INDEX]
    audio = item["audio"].squeeze(0).cpu().numpy()

    sf.write(OUT_WAV, audio, SAMPLE_RATE)
    print(f"Wrote {OUT_WAV}")
