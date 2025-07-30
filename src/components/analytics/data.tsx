export const useFetchAnalyticsTable = async (params: URLSearchParams) => {
  return await fetch(`./api/analytics-table?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

export const useFetchMasterWilayah = async (params: URLSearchParams) => {
  return await fetch(`./api/wilayah?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

export const useFecthShap = async (params: URLSearchParams) => {
  return await fetch(`./api/shap?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

export const useFetchGeoJsonLayer = async (params: URLSearchParams) => {
  return await fetch(`./api/mapservice?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};
