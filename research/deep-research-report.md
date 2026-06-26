# Synthetic Vocal Generation for Singing-Transcription Training Data: Approaches, Tooling, and Tradeoffs

## TL;DR
- **Yes, a transcription model trained on synthetic singing can come close to one trained on real data — but only if you optimize for the right thing.** The challenge scores you on Basic Pitch's COnPOff/COnP/COn F1, and the published evidence (Sato & Akama, ICASSP 2024, arXiv:2312.10402) shows annotation-free synthetic training reaches note-F1 of 0.77 on guitar vs 0.80 for full real-data supervision (−0.03) and *beats* it on orchestra (0.74 vs 0.66, +0.08). ROSVOT (ACL 2024) shows a downstream model trained purely on pseudo-labeled singing reaches 91% of human-annotated pitch accuracy. The consistent finding: **acoustic/timbre diversity matters more than per-sample realism.**
- **The winning recipe is a portfolio, not one synthesizer:** generate perfectly score-aligned audio (DiffSinger/NNSVS for quality + a deterministic WORLD/NSF f0-driven renderer for guaranteed pitch fidelity), multiply timbres cheaply via RVC/so-vits-svc voice conversion, then apply heavy domain randomization (pitch/time perturbation, reverb/RIR, codec artifacts, and — most importantly — mixing vocals with real instrumental backing tracks since Basic Pitch is instrument-agnostic and trained in-the-mix).
- **The single biggest risk is alignment drift, not audio quality.** Neural SVS models add expressive timing/vibrato that can violate the 50 ms onset tolerance. Anchor your ground truth by either driving an f0-controllable vocoder directly from the score (zero drift) or re-deriving onset/offset labels from the actually-rendered audio.

## Key Findings

**The target model is Spotify Basic Pitch.** The Klangio reference repo trains a (slightly modified) Basic Pitch — per Spotify Engineering's "Meet Basic Pitch" (June 2022), "a convolutional neural network (CNN) that has less than 20 MB peak memory and fewer than 17,000 parameters" (the precise count is 16,782) — from random init on your `(audio.wav, score.tsv)` pairs. This dictates everything: the model "expects audio at exactly 22,050 Hz, mono, as a Float32Array"; the front end is a Harmonic CQT (8 harmonic channels including one sub-harmonic, 3 bins per semitone, ~11 ms hop); outputs are three posteriorgrams (onset, note, fine-pitch/contour). Because Basic Pitch is **instrument-agnostic and was trained on in-the-mix audio** (MAESTRO, GuitarSet, MedleyDB/MDB-melody, Slakh, iKala vocals), it does not need clean solo vocals — which is exactly why mixing synthetic vocals into real backing tracks is a legitimate domain-gap closer rather than a gimmick. The challenge hint that COnP_f1 should exceed 0.15 is a floor, not a target.

**Evidence that synthetic works.** Sato & Akama (arXiv:2312.10402v3) trained a transcription Transformer on MIDI rendered with NSynth one-shot timbres — Table 1's Synthetic-DC row used **126K MIDI segments (from Lakh/LMD-full, 176,581 tracks) × 313K+ mixed timbres (from NSynth's 1,006 base timbres, mixed as T_main + αT_sub)** — plus adversarial domain confusion on unannotated real audio. Their note-level metric is defined verbatim: "a note is deemed correct if its pitch is correct, the onset error is within 50 ms, and the offset error is within 50 ms or 20 percent of the note length." Results: note-F1-no-offset of 0.77 on guitar vs 0.80 full-real (−0.03) and 0.74 vs 0.66 on orchestra (+0.08), trailing 22–33 points on piano/vocal/multitrack. Their scaling result is the key design lesson: **more timbre variation helps timbrally-diverse/vocal domains most; more MIDI/note variation helps structurally complex domains.** ROSVOT (Li et al., arXiv:2405.09940) closes the transcription↔synthesis loop: "the SVS model trained with pure transcribed annotations achieves 91% of the pitch accuracy compared to manually annotated data, without loss of overall quality." The drum-transcription literature (arXiv:2407.19823, "Analyzing and reducing the synthetic-to-real transfer gap") is the cautionary counterpoint: naïve synthetic data (low-fidelity soundfonts) transfers poorly, and realism strategies measurably narrow the gap.

## Details — Approaches Organized by Tradeoff

I rank approaches on four axes: **Quality** (naturalness), **Effort** (setup/compute), **GT-fidelity** (how perfectly the audio matches the score's onset/offset/pitch — the training signal), and **Diversity** (acoustic variety per unit effort).

### Approach A — Neural SVS toolkits (high quality, medium GT-fidelity risk)

These take score/MIDI + lyrics/phonemes and produce natural singing. Best audio realism; the catch is they inject expressive timing and pitch transitions that can drift from the literal score.

- **DiffSinger (OpenVPI fork)** — `github.com/openvpi/DiffSinger`; original `github.com/MoonInTheRiver/DiffSinger` (Liu et al., AAAI 2022, vol. 36 pp. 11020–11028, arXiv:2105.02446; the original reported "0.11 MOS gains" over prior SOTA on the PopCS Chinese singing dataset). Diffusion acoustic model, 44.1 kHz, explicit variance controls for **pitch, energy, breathiness**, gender/formant shifting, and a `--key` transpose flag — directly useful for diversity. **Batch generation is easiest via OpenUTAU** (`github.com/stakira/OpenUtau`), which has DiffSinger integration, built-in multilingual ML G2P, a vibrato editor, and exports through the **nsf_hifigan** vocoder. The DiffSinger Community Vocoders project provides a universal pretrained vocoder.
- **NNSVS** — `github.com/nnsvs/nnsvs` (Yamamoto/Yoneyama/Toda, ICASSP 2023, arXiv:2210.15987). Modular time-lag / duration / acoustic / vocoder pipeline; autoregressive F0 models capture vibrato naturally; documented support for training **universal NSF/HnSincNSF vocoders** on mixed databases. The modular F0 stage is attractive because you can override predicted F0 with score-derived F0 for fidelity.
- **VISinger 2** — `github.com/zhangyongmao/VISinger2`. VITS-based end-to-end SVS with a DSP synthesizer (DDSP-style harmonic/noise source) for fidelity; pretrained on Opencpop. Single-pass, fast.
- **Muskits / Muskits-ESPnet** — `github.com/SJTMusicTeam/Muskits` (Interspeech 2022). ESPnet-style end-to-end SVS toolkit; good for batch recipes and benchmarking multiple architectures.
- **Sinsy** — `github.com/r9y9/sinsy` (NIT, Nagoya). HMM/DNN, **takes MusicXML directly**, exposes gender factor, vibrato intensity, pitch shift. Older but rock-solid for deterministic, score-faithful batch synthesis with adjustable expressivity; supports Japanese/English/Mandarin.
- **2024–2025 SOTA (research, heavier):** **TCSinger / TCSinger 2** (arXiv:2505.14910, zero-shot multilingual style control), **TechSinger** (arXiv:2502.12572, flow-matching technique control: breathy/falsetto/glissando/vibrato), **StyleSinger**, **RMSSinger** (word-level, handles realistic scores without fine MIDI), **SmoothSinger** (arXiv:2506.21478). These maximize style/technique diversity — valuable for edge-case coverage — but are higher-effort to run at batch scale.

### Approach B — Deterministic f0-driven synthesis (lower quality, PERFECT GT-fidelity)

This is the strategically strongest lane for transcription because **the ground truth is the score itself** — you construct the f0 contour directly from the score's onset/offset/pitch, so alignment is exact by construction. Pair with vowel/voiced excitation and you get cheap, infinitely diverse, perfectly-aligned data.

- **WORLD vocoder via pyworld** — `github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder` (prebuilt: `pyworld-prebuilt`). Decompose any real voice/vowel into f0/spectral-envelope(sp)/aperiodicity(ap), **replace f0 with a score-derived contour**, resynthesize. Swapping the spectral envelope across donor voices/vowels multiplies timbres for free. This is the canonical "perfect pitch ground truth" trick.
- **Neural Source-Filter (NSF) / SiFi-GAN / PC-NSF-HiFiGAN / NSF-BigVGAN** — `github.com/nii-yamagishilab/project-NN-Pytorch-scripts` (Wang & Yamagishi), `github.com/PlayVoice/BigVGAN` (BigVGAN + NSF). These vocoders are **explicitly conditioned on F0**, so feeding the score's F0 yields high-quality audio whose pitch is guaranteed to track the score. uSFGAN/SiFi-GAN add fast, F0-controllable source-filter synthesis.
- **DDSP** — `magenta/ddsp` (Engel et al., ICLR 2020, arXiv:2001.04643) and **MIDI-DDSP** (arXiv:2112.09312). Harmonic-plus-noise synth driven by explicit f0 + loudness; differentiable, interpretable, trivially gives perfect pitch alignment and lets you extrapolate to unseen pitches. Great for "wrong-but-useful" vowel/hum timbres.
- **"Wrong but useful" excitation:** sine-plus-noise, vowel-only, or humming renderings driven by the score F0. Since Basic Pitch is timbre-agnostic and learns from harmonic structure, even crude harmonic excitation at the right f0 with the right onset envelope is a valid, perfectly-labeled training example. This is the cheapest diversity multiplier and likely the single highest-ROI quirky idea.

### Approach C — TTS → singing conversion (cheap, medium fidelity)

- **Speech-to-singing / pitch warping:** Generate spoken lyrics with any TTS, then warp to the score's pitch and timing. Classic signal-processing route is PSOLA / WORLD-based pitch correction; the I2R Speech2Singing line and the Smule "Automatic conversion of speech into song" patents document the speech-to-melody transform. Newer learned approaches: **SVPT** (arXiv:2406.02429, self-supervised speech-to-singing) and singing↔speech generative-flow work (Springer JASMP 2025). Tradeoff: timing alignment from DTW can drift, so prefer forcing the f0 contour and note boundaries directly from the score.
- **Autotune/pitch-correction as augmentation:** Deep Autotuner (Wager et al., ICASSP 2020) and canonical-time-warping pitch correction can be used both to *create* singing from speech and to *snap* a slightly-off synthetic performance back onto the score grid before labeling.

### Approach D — Voice conversion as a timbre multiplier (best diversity-per-effort)

This is how you cheaply turn one good performance into dozens of singers — directly attacking the "timbre variation helps most" finding.

- **RVC** — `github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`. Fast, pitch-and-speaker-conditioned VITS-based SVC; the AI Harmonizer paper (arXiv:2506.18143) uses exactly RVC + Basic Pitch together. Feature-retrieval index improves timbre stability.
- **so-vits-svc** — `github.com/svc-develop-team/so-vits-svc` (4.0), `github.com/PlayVoice/whisper-vits-svc` (5.0, BigVGAN). SoftVC content encoder + F0 → VITS + NSF-HiFiGAN; **conserves pitch and intonation through conversion** (critical: VC preserves your score-aligned f0 while changing timbre). Use `auto_predict_f0=False` so pitch is not re-predicted.
- Workflow: render once with DiffSinger/WORLD, then fan out to N RVC/so-vits-svc speaker models → N timbres with identical alignment. The challenge explicitly allows multiple `score_NNN_singer_X` pairs per score.

### Approach E — Full-song / accompaniment generation (domain-gap closer)

- **ACE-Step** — `github.com/ace-step/ACE-Step` (arXiv:2506.00045) and **ACE-Step 1.5** (arXiv:2602.00744). Open foundation music model with **lyric2vocal and singing2accompaniment** sub-tasks and vocal-to-BGM conversion. Useful for generating realistic backing tracks to mix under your aligned vocals, or as a diversity source — but it does NOT give score-aligned ground truth on its own, so use it for accompaniment/mix realism, not as the labeled vocal source.
- **Mixing synthetic vocals with real instrumental backing tracks** is the most defensible domain-gap closer: Basic Pitch was trained in-the-mix, and the MIR-ST500 pipeline itself runs Spleeter/source separation on YouTube pop. Add real or generated accompaniment under your perfectly-aligned vocal, keeping labels tied to the vocal.

### Cross-cutting: Data augmentation & domain randomization (do this regardless of synth choice)

The MIR literature is unanimous that diversity > realism for transfer. Apply, with labels preserved:
- **Pitch shift / time stretch** (small, ±semitones / ±10%), **speed perturbation** (0.9/1.0/1.1).
- **Reverb / room impulse responses** — convolve dry synthetic vocals with RIRs; **ReverbFX** (arXiv:2505.20533) provides plugin-derived RIRs specifically for singing; MUSAN for noise.
- **Codec artifacts** (MP3/Opus/AAC re-encoding), mic/device IRs, EQ, dynamic-range compression — pop vocals are heavily processed, so simulate the production chain.
- **Backing-track mixing** at varied SNRs (see Approach E).
- Libraries: `audiomentations`, `pedalboard` (Spotify's own DSP plugin host — convenient since the judge is Spotify's model), `torch-audiomentations`, `librosa`, `torchaudio`.

### Lyrics & alignment tooling

- **LLM-generated lyrics constrained to syllable counts** matching note counts per phrase: prompt an LLM (or use a syllable counter like `pyphen`/`cmudict`) to emit lyrics with exactly one syllable per note. This is the natural creative extension and gives linguistic diversity for free. For melismas (one syllable over multiple notes), assign slurs.
- **G2P / phonemizers:** `phonemizer` (espeak-ng backend), CMUdict, `pypinyin` (Mandarin), OpenUTAU's built-in ML G2P. DiffSinger/NNSVS need phoneme + duration inputs.
- **Montreal Forced Aligner (MFA)** — pretrained acoustic + G2P models (Pynini wFST G2P); use to align lyrics to rendered audio and to **re-derive note boundaries from the actually-rendered audio** when using expressive neural SVS (closes the alignment-drift gap).
- **Score/MIDI utilities:** `pretty_midi`, `music21`, `miditoolkit` to convert the `.tsv` (onset, offset, MIDI pitch) into MIDI/MusicXML for Sinsy/OpenUTAU; `mido`.
- **F0 / analysis:** `librosa`, `praat-parselmouth`, `pyworld`, CREPE/`torchcrepe`, `pyin` — for verifying your rendered audio actually tracks the score before you commit it to the training set.

### Datasets to mine for real timbres, donor voices, and validation

- **Singing-transcription (note-level, your task):** MIR-ST500 (`github.com/york135/singing_transcription_ICASSP2021`, 500 pop songs, the closest analog to Klangio's hidden set), TONAS, SSVD, DALI (large but noisy labels), ISMIR2014.
- **SVS corpora (for synthesis training / VC donor voices):** Opencpop (Mandarin, 5.2 h), M4Singer (20 singers, SATB, NeurIPS 2022), OpenSinger, **GTSinger** (NeurIPS 2024, 80.59 h, 9 languages, 20 singers, 6 techniques — best for diversity), NUS-48E, NHSS, CSD (Children's Song Dataset), Tohoku Kiritan, PJS, JVS-MuSiC, ACE-Opencpop/ACE-KiSing (scaled-up), SingNet (arXiv:2505.09325, in-the-wild).

## Recommendations

**Stage 1 — Baseline that cannot drift (day 1).** Build the deterministic f0-driven renderer first: parse `.tsv` → step/glide F0 contour → WORLD or NSF/DDSP synthesis with vowel excitation. This guarantees COnP_f1 well above the 0.15 floor because pitch and onsets are exact by construction. Validate with `torchcrepe` that rendered f0 matches the score within Basic Pitch's tolerance. Benchmark: get a non-trivial COnPOff_f1 on `klangiodataset` before adding anything.

**Stage 2 — Add naturalness and timbre diversity (days 2–3).** Layer in DiffSinger (via OpenUTAU batch) and/or NNSVS for realistic vocals, then fan out every rendered take through 5–20 RVC / so-vits-svc speaker models (`auto_predict_f0=False`). Generate LLM lyrics with syllable-count constraints. Add vibrato/portamento/breathiness via the synth's variance controls for edge-case coverage. **Threshold to watch:** if neural-SVS samples drop COn_f1 relative to the deterministic baseline, you have alignment drift — fix by re-deriving labels via MFA or by forcing score F0 into the vocoder.

**Stage 3 — Close the domain gap (days 3–4).** This is where you most likely beat the real-data baseline. Mix every clean vocal under real or ACE-Step-generated accompaniment at varied SNRs; convolve with ReverbFX RIRs; apply MP3/Opus codec round-trips and pedalboard production-chain effects. Because Basic Pitch is instrument-agnostic and trained in-the-mix, in-the-mix synthetic data is the highest-value augmentation. **Threshold:** monitor COnPOff_f1 on `klangiodataset`; if mixed-in data hurts, reduce accompaniment level or ensure the vocal remains the dominant harmonic source.

**Stage 4 — Scale by the right axis.** Per Sato & Akama, for a vocal/timbrally-diverse target, **spend your remaining budget on timbre variation (more VC speakers, more reverb/codec/mix conditions), not on more unique scores.** Generate multiple augmented variants per score.

**What would change these recommendations:** If the deterministic renderer alone already matches the hidden baseline (you'll only learn this at final scoring), skip neural SVS entirely — it adds risk. If COn_f1 is high but COnPOff_f1 lags, your offsets are wrong: tighten note-release/decay envelopes. If COnP_f1 is high but generalization to the hidden set is poor, you're under-diversified — add more timbre/mix randomization.

## Caveats

- **Alignment is the dominant risk.** Expressive neural SVS (vibrato, portamento, pre-onset consonants, timing humanization) can push note starts beyond the 50 ms tolerance. The deterministic/f0-forced lane sidesteps this entirely; for neural SVS, re-derive labels from rendered audio.
- **Some Basic Pitch front-end specifics are inferred.** The 3-bins-per-semitone HCQT, 8-channel harmonic stack, ~11 ms hop, and 16,782-parameter count are corroborated across multiple sources, but the exact hop in samples and the shipped model's full training set are not quotable verbatim from the primary 2022 paper PDF (the README warns the shipped model improves on the paper). Treat front-end details as design guidance, not gospel.
- **The synthetic-to-real gap is real on data-rich domains.** Sato & Akama still trail full real supervision by 22–33 F1 points on piano and vocal mixtures; the strong wins are on guitar/orchestra. Singing in-the-mix is closer to the vocal/Slakh regime, so expect the gap to be non-trivial — closing it is precisely the challenge, and timbre+mix diversity is the lever. Note also that the ROSVOT "91%" figure refers to a downstream SVS model trained on its pseudo-labels, not directly to a Basic-Pitch-style note transcriber.
- **Licensing/ethics:** RVC/so-vits-svc and SVS repos carry explicit prohibitions on cloning real identifiable people without consent; use synthetic or properly-licensed donor voices. GTSinger and the SVS corpora have research licenses — check terms before redistributing generated data.
- **Compute:** DiffSinger/ACE-Step need a GPU; WORLD/DDSP/pyworld run fast on CPU. The deterministic lane is the most laptop-friendly and the most defensible — favor it under time pressure.