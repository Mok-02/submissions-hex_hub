# SIS — Satellite Intelligence System

## Problem and Customer Value

Satellite datasets like EuroSAT_MS contain thousands of images showing land types such as agriculture, forests, urban areas, and water bodies. In real-world situations like floods or droughts, people do not need raw images — they need clear answers.

For example, a question like “Which of these 60 tiles show flooding?” should not require a human to manually inspect each image. This process can take hours and requires trained analysts.

SIS is designed for emergency response teams that receive satellite data but may not have experts available to analyze it quickly. The system converts a time-consuming manual task into an automated one that runs in milliseconds.

Instead of sending large image files (around 42 MB for 60 tiles), SIS processes the data on the system and returns a small structured output of about 500 bytes. This reduces bandwidth usage by nearly 99.99% while still providing useful insights such as the number of affected areas, their locations, and confidence scores.

---

## What We Built

SIS works as a pipeline with four main stages.

First, the system understands the user’s query. A lightweight natural language parser extracts intent from simple phrases like “flood risk” or “crop stress.” This step is fast and does not rely on large AI models.

Next, each satellite image is converted into a numerical representation. We generate a 768-dimensional vector for every tile, capturing important features like vegetation health (NDVI), water presence, and land type.

Then, the system ranks all tiles using a combination of similarity matching and rule-based scoring. Similarity measures how close an image is to the query, while rules use domain knowledge such as NDVI thresholds and water levels to improve accuracy.

Finally, the system generates a structured answer. Instead of returning images, it outputs useful information like matching tiles, their coordinates, and confidence scores in a compact format.

---

## Dataset and TerraMind Integration

We use the EuroSAT_MS dataset, which contains 27,000 satellite images with 13 spectral bands. For demonstration purposes, we work with a smaller sample of 60 tiles across different land types such as agriculture, urban, water, and forest.

We also integrated TerraMind-tiny into our system. This model processes satellite inputs and generates feature representations. However, without fine-tuning, we observed that TerraMind’s raw features do not clearly separate different land types. Because of this, we rely on engineered spectral features like NDVI and water indices to achieve better performance in our current version.

---

## Performance

SIS achieves a major reduction in bandwidth by replacing large image transfers with compact structured outputs. A typical query that would require 42 MB of image data is reduced to around 500 bytes.

In terms of accuracy, our engineered feature approach performs significantly better than raw TerraMind embeddings. It produces clearer separation between land types and consistently returns correct results for test queries.

The system is also fast. It processes a query in approximately 15 to 75 milliseconds on a standard CPU, without requiring a GPU.

---

## Space Compute Fit

SIS is designed with space constraints in mind. It avoids heavy models and instead uses efficient, deterministic methods that require minimal power and memory. The system runs quickly, produces consistent results, and minimizes the amount of data that needs to be transmitted.

This design choice is intentional. In environments like satellites, where power and bandwidth are limited, simpler and more reliable methods are often more practical than large neural networks.

---

## Limitations and Future Work

There are still areas for improvement. TerraMind has not yet been fine-tuned, which limits its effectiveness for distinguishing land types. The current system also uses a small sample of tiles, while a real deployment would need to scale to thousands of images using optimized search methods.

The system currently analyzes only a single snapshot of data. Many real-world problems, such as flood detection, would benefit from comparing images over time. Additionally, the query parser is basic and relies on keywords, so it may not handle more complex or varied language.

Finally, the system has not yet been tested on actual satellite hardware, and performance estimates are based on local testing.

---

## Conclusion

SIS transforms satellite imagery into actionable insights. It reduces analysis time from hours to milliseconds, minimizes bandwidth usage, and provides clear, structured answers instead of raw data. This makes it a practical and efficient solution for real-time decision-making in resource-constrained environments.
