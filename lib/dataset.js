/**
 * lib/dataset.js
 *
 * Loads the AG News dataset directly from Hugging Face by URL
 * (no manual download / no local CSV) using the public
 * datasets-server "rows" REST API, then filters it down to the
 * Sci/Tech label only (label id = 3 in ag_news).
 *
 * Dataset card: https://huggingface.co/datasets/ag_news
 * API docs:     https://huggingface.co/docs/datasets-server
 */

const HF_ROWS_ENDPOINT = 'https://datasets-server.huggingface.co/rows';
const DATASET = 'ag_news';
const CONFIG = 'default';
const SCI_TECH_LABEL_ID = 3; // 0=World, 1=Sports, 2=Business, 3=Sci/Tech
const PAGE_SIZE = 100; // max allowed by the datasets-server API per request

/**
 * Fetch one page of raw AG News rows from Hugging Face.
 */
async function fetchPage(split, offset) {
  const url = `${HF_ROWS_ENDPOINT}?dataset=${DATASET}&config=${CONFIG}&split=${split}&offset=${offset}&length=${PAGE_SIZE}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Hugging Face datasets-server error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

/**
 * AG News rows look like: { row: { text: "Title \\ short description", label: 3 } }
 * The title and the short body are joined with a backslash in the raw field,
 * so we split them back apart for nicer cards/citations in the UI.
 */
function splitTitleAndBody(rawText) {
  const parts = rawText.split(' \\ ');
  if (parts.length >= 2) {
    return { title: parts[0].trim(), body: parts.slice(1).join(' \\ ').trim() };
  }
  return { title: rawText.slice(0, 80), body: rawText };
}

/**
 * Pull up to `limit` Sci/Tech-labeled articles from AG News.
 * Paginates through the HF datasets-server API and filters client-side,
 * since the API does not support server-side label filtering.
 *
 * @param {number} limit max number of Sci/Tech articles to return
 * @param {string} split "train" (120k rows) or "test" (7.6k rows)
 */
async function loadSciTechArticles(limit = 300, split = 'test') {
  const articles = [];
  let offset = 0;
  const maxOffset = split === 'train' ? 120000 : 7600;

  while (articles.length < limit && offset < maxOffset) {
    const page = await fetchPage(split, offset);
    const rows = page.rows || [];
    if (rows.length === 0) break;

    for (const { row } of rows) {
      if (row.label === SCI_TECH_LABEL_ID) {
        const { title, body } = splitTitleAndBody(row.text);
        articles.push({
          id: `ag-scitech-${split}-${offset}-${articles.length}`,
          title,
          body,
          text: row.text,
          label: 'Sci/Tech',
        });
        if (articles.length >= limit) break;
      }
    }
    offset += PAGE_SIZE;
  }

  return articles;
}

module.exports = { loadSciTechArticles };
