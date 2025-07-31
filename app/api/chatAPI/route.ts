import { useSendChat } from '@/components/analytics/data';
import { getResponseBody, ResponseBody } from '@/types/response-body';
import { ChatBody } from '@/types/types';

export const runtime = 'edge';

export async function POST(req: Request): Promise<ResponseBody> {
  const responseBody:ResponseBody = getResponseBody();
  try {
    const requestBody = (await req.json()) as ChatBody;
    const response = await useSendChat(requestBody);
    if (response.success) {
      console.log(response);
      responseBody.data = response;
      return response
    } else {
      throw Error(response.message);
    }

  } catch (error:any) {
    responseBody.success = false;
    responseBody.message = error.message;
    return responseBody;
  }
}
