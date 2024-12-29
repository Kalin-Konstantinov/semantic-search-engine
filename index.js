import 'dotenv/config';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { Chroma } from "@langchain/community/vectorstores/chroma";


const loader = new PDFLoader("./news.pdf");

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

const embeddings = new OllamaEmbeddings({
    // apiKey: process.env.OPENAI_API_KEY,
    model: "mxbai-embed-large:latest"
});

const vectorStore = new Chroma(embeddings, {
    collectionName: "a-new-test-collection2",
    url: "http://0.0.0.0:8000",
});

await vectorStore.addDocuments(allSplits);

const results2 = await vectorStore.similaritySearchWithScore(
    "What Purdue SAFE-RWSL surveillance system built to prevent airport runway incursions?"
);


console.log(results2[0]);
