export type OpenAIModel = 'gpt-4o' | 'gpt-3.5-turbo';

export interface ChatBody {
  message: string;
  session_id: string;
  include_analysis: boolean;
}
