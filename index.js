import 'dotenv/config';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const loader = new PDFLoader("./nke-10k-2023.pdf");

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

const embeddings = new OllamaEmbeddings({
    model: "mxbai-embed-large",
    baseUrl: process.env.OLLAMA_BASE_URL,
});

const vectorStore = new MemoryVectorStore(embeddings);
await vectorStore.addDocuments(allSplits);

const results1 = await vectorStore.similaritySearch(
    "When was Nike incorporated?"
);

console.log(results1[0]);
