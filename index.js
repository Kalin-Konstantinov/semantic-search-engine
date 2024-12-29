import 'dotenv/config';
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";


const loader = new PDFLoader("./nke-10k-2023.pdf");

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

const embeddings = new OpenAIEmbeddings({
    apiKey: process.env.OPENAI_API_KEY,
    model: "text-embedding-3-large"
});

const vectorStore = new Chroma(embeddings, {
    collectionName: "a-test-collection",
    url: "http://0.0.0.0:8000",
});

await vectorStore.addDocuments(allSplits);

const results2 = await vectorStore.similaritySearchWithScore(
    "What was Nike's revenue in 2023?"
);


console.log(results2[0]);
