import { useSendChat } from '@/components/analytics/data';
import { getResponseBody, ResponseBody } from '@/types/response-body';
import { ChatBody } from '@/types/types';
import { NextResponse } from 'next/server';

export const runtime = 'edge';

export async function POST(req: Request): Promise<Response> {
  const responseBody:ResponseBody = getResponseBody();
  try {
    const requestBody = (await req.json()) as ChatBody;
    const response = await useSendChat(requestBody);
    if (response.success) {
      responseBody.data = response;
      return NextResponse.json(responseBody);
    } else {
      throw Error(response.message);
    }

  } catch (error:any) {
    responseBody.success = false;
    responseBody.message = error.message;
    return NextResponse.json(responseBody);
  }
}
