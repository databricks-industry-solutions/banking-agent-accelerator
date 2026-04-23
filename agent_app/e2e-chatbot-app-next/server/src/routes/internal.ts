import { Router, type Request, type Response, type Router as RouterType } from 'express';
import { updateChatWorkflowState, isDatabaseAvailable } from '@chat-template/db';

export const internalRouter: RouterType = Router();

/**
 * POST /api/internal/background-check-received
 *
 * Called by send_background_check.py (or any external system) after injecting
 * a background-check result into the LangGraph checkpoint. Updates the chat's
 * stage in the DB so the sidebar reflects that a result is ready before the
 * user sends their next message.
 *
 * No auth required -- intended for localhost/internal use only.
 */
internalRouter.post(
  '/background-check-received',
  async (req: Request, res: Response) => {
    const { chatId } = req.body;

    if (!chatId || typeof chatId !== 'string') {
      return res.status(400).json({ error: 'chatId is required' });
    }

    if (!isDatabaseAvailable()) {
      return res.status(503).json({ error: 'Database not available' });
    }

    try {
      await updateChatWorkflowState({
        chatId,
        stage: 'BACKGROUND_CHECK_RECEIVED',
      });
      console.log(`[Internal] Background check received for chat ${chatId}`);
      return res.status(200).json({ success: true });
    } catch (err) {
      console.error('[Internal] Failed to update workflow state:', err);
      return res.status(500).json({ error: 'Failed to update chat state' });
    }
  },
);
