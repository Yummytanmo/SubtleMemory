# Related 10 Diverse Personas

This file documents the current 10-persona selection used by
`build-config.related-10-diverse-personas.json`.

## Final Persona IDs

```json
[
  0,
  334,
  351,
  341,
  92,
  637,
  45,
  996,
  135,
  144
]
```

Compared with the previous selection:

- `126` was replaced by `0`.
- `388` was replaced by `92`.

The first-persona test uses the same config file with `--limit 1`, so the
test and full build share the same run directory and config.

## Personas

| persona_id | Persona | Coverage |
| --- | --- | --- |
| `0` | Amara Nwosu, a Nigerian-American STEM education program designer with physics and mechanical engineering training, known for turning physics concepts into hands-on toy car race activities. | STEM education, physics, mechanical engineering, youth outreach |
| `334` | Dr. Suthida Rattanaporn, a Thai nurse educator dedicated to improving nursing curricula and training methods. | Healthcare, nursing education, curriculum design |
| `351` | Gareth McAllister, an Australian family farmer managing stress and uncertainty around farm operations. | Agriculture, rural work, family business, stress management |
| `341` | Malik Thompson, an African American artist in Brooklyn with strong political and social expression. | Art, politics, social justice, urban culture |
| `92` | Julian Marcus Ortega, a computational sociologist studying social media dynamics with NLP over large user-generated datasets. | AI-adjacent research, NLP, computational social science, social media |
| `637` | Lars Henriksen, a Norwegian professional biathlete preparing for competition. | Professional sports, endurance training, performance optimization |
| `45` | Isabel Moraes Santiago, an elderly Oliventine Portuguese speaker with a strong connection to cultural heritage and language preservation. | Elderly persona, cultural heritage, language preservation |
| `996` | Christopher "Chris" Halberg, a disbarred former New Zealand prosecutor with insider knowledge of legal misconduct. | Law, institutional accountability, professional fallibility |
| `135` | Adriana Ionescu, a Romanian migrant living in Hungary and navigating cross-cultural identity. | Migration, cross-cultural adaptation, urban planning background |
| `144` | Dr. Amira Suryani, an Indonesian neuroscientist studying mind uploading and consciousness transfer. | Neuroscience, speculative technology, ethics, consciousness research |

## Diversity Rationale

This set is intended to cover:

- Geographic and cultural spread: Nigerian-American, Thai, Australian, African American, Norwegian, Oliventine Portuguese, New Zealander, Romanian migrant in Hungary, Indonesian.
- Age range: teen/young-adult-adjacent STEM outreach through elderly cultural preservation.
- Domains: STEM education, healthcare, agriculture, art and politics, NLP/social science research, elite sports, language heritage, law, migration, neuroscience.
- Preference styles: each source profile includes stereotypical, anti-stereotypical, and neutral preferences, supporting relation-aware memory construction beyond simple demographic stereotypes.

## Build Commands

First-persona test build:

```bash
bash scripts/generate_related_10_personas_first1.sh
```

Full 10-persona build:

```bash
bash scripts/generate_related_10_personas.sh
```

Run test first, then full:

```bash
bash scripts/generate_related_10_personas_pipeline.sh
```

All commands use the same config file,
`construction/user-related/examples/build-config.related-10-diverse-personas.json`,
the same output base directory, `data/user-related`, and the same run id,
`related-10-diverse-personas`. The first command adds `--limit 1`; the full
command omits `--limit` and expands the same run to the full configured
persona list.
