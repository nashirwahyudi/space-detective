import { NextResponse } from 'next/server';
import { pool } from '@/lib/db';
import { getResponseBody, ResponseBody } from '@/types/response-body';

export async function GET(req: Request) {
  const response: ResponseBody = getResponseBody();
  try {
    const { searchParams } = new URL(req.url);

    const h3Index = searchParams.get('h3Index') || null;

    if (!h3Index) {
      throw Error('Parameter h3Index must exist.');
    }

    let query = `SELECT * FROM plot_metadata WHERE h3_index=$1 order by index desc`;
    const { rows } = await pool.query(query, [h3Index]);
    response.data = rows;

    return NextResponse.json(response);
  } catch (err: any) {
    response.success = false;
    response.message = err.message || 'Terjadi kesalahan!';
    return NextResponse.json(response);
  }
}
