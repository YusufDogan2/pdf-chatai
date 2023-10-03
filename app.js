import { OpenAI } from "langchain/llms/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";

import * as dotenv from 'dotenv';
import express from 'express';
dotenv.config();




const app = express();
const port = 9898;


app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});


app.get('/ask', async (req, res) => {
  try {

    const llmA = new OpenAI({ modelName: "gpt-3.5-turbo" });
    const chainA = loadQAStuffChain(llmA);
    const directory = "C:\Users\yusuf\Desktop\pdf-hack"; //saved directory in .env file

    const loadedVectorStore = await FaissStore.load(
      directory,
      new OpenAIEmbeddings()
    );

    const question = "Können Pflegeeltern die leiblichen Eltern ersetzen?";
    const result = await loadedVectorStore.similaritySearch(question, 1);
    const resA = await chainA.call({
      input_documents: result,
      question,
    });
    console.log({ resA });
    res.json({ result: resA });
  }

  catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});