import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { getResponseBody, ResponseBody } from '@/types/response-body';

export async function GET(req: Request) {
  const response: ResponseBody = getResponseBody();
  try {
    let query = `SELECT * FROM plot_metadata WHERE 1=1 order by index desc`;
    const { rows } = await pool.query(query);
    response.data = rows;

    return NextResponse.json(response);
  } catch (err: any) {
    response.success = false;
    response.message = err.message || 'Terjadi kesalahan!';
    return NextResponse.json(response);
  }
}
