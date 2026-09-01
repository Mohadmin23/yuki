async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // Keep the HTTP fallback.
    }
    throw new Error(detail);
  }
  return response.json();
}

export const api = {
  dataset: () => request("/api/dataset"),
  sample: (index) => request(`/api/samples/${index}`),
  navigate: ({ current, direction, verdict, search }) => {
    const query = new URLSearchParams({ current, direction, verdict, search });
    return request(`/api/navigate?${query}`);
  },
  saveReview: (payload) => request("/api/reviews", {
    method: "POST",
    body: JSON.stringify(payload),
  }),
  saveDraft: (payload) => request("/api/drafts", {
    method: "PUT",
    body: JSON.stringify(payload),
  }),
  undo: () => request("/api/reviews/undo", { method: "POST" }),
  dashboard: () => request("/api/dashboard"),
  history: () => request("/api/reviews/history?limit=200"),
  exports: () => request("/api/exports", { method: "POST" }),
  clearReviews: (confirmation) => request("/api/reviews/clear", {
    method: "POST",
    body: JSON.stringify({ confirmation }),
  }),
};
