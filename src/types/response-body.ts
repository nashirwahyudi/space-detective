import { time } from 'console';

export interface ResponseBody {
  success: boolean;
  data: object | null;
  message: string;
  issued: Date;
}

export const getResponseBody = () => {
  const response: ResponseBody = {
    success: true,
    message: 'Berhasil mengambil data',
    issued: new Date(),
    data: null,
  };
  return response;
};
