export const useFetchAnalyticsTable = async (params: URLSearchParams) => {
  return await fetch(`./api/backend?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  })
    .then(async (res) => await res.json())
};
