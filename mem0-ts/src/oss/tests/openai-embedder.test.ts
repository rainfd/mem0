/// <reference types="jest" />

import { OpenAIEmbedder } from "../src/embeddings/openai";

const mockCreate = jest.fn();

jest.mock("openai", () => {
  return jest.fn().mockImplementation(() => ({
    embeddings: { create: mockCreate },
  }));
});

describe("OpenAIEmbedder (unit)", () => {
  beforeEach(() => {
    mockCreate.mockReset();
  });

  it("passes dimensions when embeddingDims is configured", async () => {
    mockCreate.mockResolvedValueOnce({
      data: [{ embedding: [0.1, 0.2, 0.3] }],
    });

    const embedder = new OpenAIEmbedder({
      apiKey: "test-key",
      baseURL: "https://example.test/v1",
      model: "text-embedding-v4",
      embeddingDims: 1536,
    });

    await embedder.embed("hello");

    expect(mockCreate).toHaveBeenCalledWith({
      model: "text-embedding-v4",
      input: "hello",
      encoding_format: "float",
      dimensions: 1536,
    });
  });

  it("does not pass dimensions when embeddingDims is not configured", async () => {
    mockCreate.mockResolvedValueOnce({
      data: [{ embedding: [0.1, 0.2, 0.3] }],
    });

    const embedder = new OpenAIEmbedder({
      apiKey: "test-key",
      baseURL: "https://example.test/v1",
      model: "text-embedding-3-small",
    });

    await embedder.embed("hello");

    expect(mockCreate).toHaveBeenCalledWith({
      model: "text-embedding-3-small",
      input: "hello",
      encoding_format: "float",
    });
  });
});
