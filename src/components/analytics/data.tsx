import { ChatBody } from "@/types/types";

export const useFetchAnalyticsTable = async (params: URLSearchParams) => {
  return await fetch(`/api/analytics-table?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

export const useFetchMasterWilayah = async (params: URLSearchParams) => {
  return await fetch(`/api/wilayah?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

export const useFecthShap = async (params: URLSearchParams) => {
  return await fetch(`/api/shap?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

export const useFetchGeoJsonLayer = async (params: URLSearchParams) => {
  return await fetch(`/api/mapservice?${params.toString()}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  }).then(async (res) => await res.json());
};

// chat api
export const useSendChat = async (requestBody: ChatBody) => {
  const controller = new AbortController();
  return await fetch(`${process.env.NEXT_PUBLIC_CHAT_API}`, {
    method: 'POST',
    body: JSON.stringify(requestBody),
    mode: 'cors',
    headers: {
      'Content-Type': 'application/json'
    },
    signal: controller.signal
  }).then(async (res) => await res.json());
}