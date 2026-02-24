import { GoogleGenAI } from "@google/genai";
import { UserMemory, StudyPlan, Resource, Question } from '../types';
import { retrieveRelevantChunks } from './localEmbeddingService';


const getClient = () => {
  const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("VITE_GEMINI_API_KEY is not set in environment variables");
  }
  return new GoogleGenAI({ apiKey });
};

// --- Helper: Clean JSON ---
const cleanAndParseJSON = (text: string) => {
  try {
    // Remove Markdown code blocks
    let cleaned = text.replace(/```json/g, '').replace(/```/g, '');
    
    // Find valid JSON bounds
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    const firstBracket = cleaned.indexOf('[');
    const lastBracket = cleaned.lastIndexOf(']');

    // Determine if we are looking for object or array
    if (firstBrace !== -1 && lastBrace !== -1 && (firstBracket === -1 || firstBrace < firstBracket)) {
      cleaned = cleaned.substring(firstBrace, lastBrace + 1);
    } else if (firstBracket !== -1 && lastBracket !== -1) {
      cleaned = cleaned.substring(firstBracket, lastBracket + 1);
    }
    
    return JSON.parse(cleaned);
  } catch (e) {
    console.error("JSON Parse Error", e);
    return null;
  }
};

// --- Context Injection ---
const constructSystemInstruction = (memory: UserMemory, contextType: string) => {
  return `
    Bạn là AI Mentor, một trợ lý học tập AI thông minh, điềm tĩnh và chuyên nghiệp.
    
    THÔNG TIN NGƯỜI DÙNG:
    Tên: ${memory.name}
    Lớp: ${memory.grade} (Hệ thống giáo dục Việt Nam)
    
    VAI TRÒ CỦA BẠN:
    Hiện tại bạn đang đóng vai trò: ${contextType}.
    
    NGUYÊN TẮC:
    1. Ngôn ngữ: 100% Tiếng Việt.
    2. Nội dung: Chính xác, phù hợp giáo dục Việt Nam.
    3. Khi có công thức toán, luôn viết đúng Markdown + LaTeX:
       - Inline: $...$
       - Xuống dòng riêng: $$...$$
       - Không escape ký tự $.
  `;
};

// --- Feature: Quiz Generation ---
export const generateQuizForTask = async (taskDescription: string, memory: UserMemory): Promise<Question | null> => {
  const ai = getClient();
  const prompt = `
    Dựa trên nhiệm vụ học tập: "${taskDescription}".
    Hãy tạo 1 câu hỏi trắc nghiệm (Multiple Choice) để kiểm tra xem học sinh đã hiểu bài chưa.
    Trình độ: ${memory.grade}.
    
    Trả về JSON duy nhất (không markdown):
    {
      "id": "q1",
      "text": "Nội dung câu hỏi?",
      "options": ["Đáp án A", "Đáp án B", "Đáp án C", "Đáp án D"],
      "correctAnswer": 0,
      "explanation": "Giải thích ngắn gọn tại sao đáp án này đúng."
    }
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        systemInstruction: constructSystemInstruction(memory, 'Giáo viên Kiểm tra'),
      }
    });

    const text = response.text || '';
    return cleanAndParseJSON(text) as Question;
  } catch (error) {
    console.error("Quiz Gen Error:", error);
    return null;
  }
};

// --- Feature: Study Plan Generator ---
export const generateStudyPlan = async (subject: string, duration: string, memory: UserMemory): Promise<StudyPlan | null> => {
  const ai = getClient();
  const prompt = `
    Lập lộ trình học tập chi tiết cho môn "${subject}" trong thời gian "${duration}".
    Học sinh đang học ${memory.grade}.
    Các phần kiến thức học sinh còn yếu: ${memory.weaknesses.join(', ')}.
    
    Yêu cầu:
    - Bám sát chương trình ${memory.grade} của Việt Nam.
    - Chia nhỏ nhiệm vụ cụ thể.
    
    Trả về định dạng JSON CHÍNH XÁC theo mẫu sau (không thêm markdown code block, không thêm lời dẫn):
    {
      "subject": "${subject}",
      "duration": "${duration}",
      "weeks": [
        {
          "week": 1,
          "days": [
            {
              "day": 1,
              "tasks": [
                { "id": "t1", "description": "Tên nhiệm vụ cụ thể (VD: Ôn tập chương 1 SGK)", "completed": false }
              ]
            }
          ]
        }
      ]
    }
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        systemInstruction: constructSystemInstruction(memory, 'Chuyên gia Lập kế hoạch Giáo dục'),
      }
    });

    const text = response.text || '';
    return cleanAndParseJSON(text) as StudyPlan;
  } catch (error) {
    console.error("Gemini Plan Error:", error);
    return null;
  }
};

// --- Feature: Resource Discovery ---
export const discoverResources = async (subject: string, memory: UserMemory): Promise<Resource[]> => {
  const ai = getClient();
  const prompt = `
    Tìm 5 tài liệu học tập trực tuyến chất lượng cao, miễn phí cho môn "${subject}" trình độ ${memory.grade} tại Việt Nam.
    Ưu tiên các nguồn: VietJack, Hocmai, OLM, Vuihoc, hoặc kênh Youtube giáo dục uy tín của Việt Nam.
    
    Trả về định dạng JSON (không markdown):
    [
      {
        "id": "unique_id",
        "title": "Tên tài liệu",
        "type": "video" | "article" | "exercise",
        "url": "https://example.com",
        "authority": "High",
        "description": "Mô tả ngắn gọn về nội dung."
      }
    ]
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        systemInstruction: constructSystemInstruction(memory, 'Thủ thư Học thuật'),
      }
    });

    const text = response.text || '';
    return cleanAndParseJSON(text) || [];
  } catch (error) {
    console.error("Resource Discovery Error:", error);
    return [];
  }
};

// --- Feature: Scriba (PDF Chat) ---
export const chatWithScriba = async (
  message: string, 
  history: {role: string, parts: {text: string}[]}[], 
  documentContext: string,
  memory: UserMemory
): Promise<string> => {
  const ai = getClient();
  const relevantChunks = retrieveRelevantChunks(message, documentContext, {
    topK: 6,
    maxChunkChars: 1100,
    overlapChars: 180,
  });
  const groundedContext =
    relevantChunks.length > 0
      ? relevantChunks.join('\n\n---\n\n')
      : documentContext.substring(0, 12000);
  
  const chat = ai.chats.create({
    model: 'gemini-3-flash-preview',
    history: history,
    config: {
      systemInstruction: `
        ${constructSystemInstruction(memory, 'Trợ lý Tài liệu Scriba')}
        
        BỐI CẢNH TÀI LIỆU (đã truy hồi bằng local embedding chạy tại máy người dùng):
        ${groundedContext}
        
        Hãy trả lời dựa trên bối cảnh phía trên. Giải thích dễ hiểu, phù hợp với học sinh ${memory.grade}.
        Nếu cần viết công thức, bắt buộc dùng LaTeX markdown để hiển thị đúng.
      `,
    }
  });

  try {
    const result = await chat.sendMessage({ message });
    return result.text || "Xin lỗi, tôi không thể xử lý yêu cầu này.";
  } catch (error) {
    console.error("Scriba Error:", error);
    return "Tôi đang gặp khó khăn khi đọc tài liệu lúc này.";
  }
};

// --- Feature: AI Notes ---
export const enhanceNote = async (content: string, action: 'summarize' | 'simplify' | 'quiz', memory: UserMemory): Promise<string> => {
  const ai = getClient();
  let prompt = "";
  
  switch(action) {
    case 'summarize': prompt = "Tóm tắt ghi chú này thành 3 ý chính quan trọng nhất."; break;
    case 'simplify': prompt = "Giải thích khái niệm này một cách đơn giản nhất, lấy ví dụ thực tế."; break;
    case 'quiz': prompt = "Tạo 3 câu hỏi trắc nghiệm ôn tập dựa trên ghi chú này (có đáp án)."; break;
  }

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Nội dung: "${content}"\n\nYêu cầu: ${prompt}\n\nNếu có công thức toán thì dùng chuẩn markdown LaTeX: inline $...$, block $$...$$.`,
      config: {
        systemInstruction: constructSystemInstruction(memory, 'Gia sư Riêng'),
      }
    });
    return response.text || "";
  } catch (error) {
    return "Không thể xử lý ghi chú lúc này.";
  }
};
