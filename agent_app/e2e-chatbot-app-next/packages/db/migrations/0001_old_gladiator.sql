ALTER TABLE "ai_chatbot"."Chat" ADD COLUMN "stage" varchar(64);--> statement-breakpoint
ALTER TABLE "ai_chatbot"."Chat" ADD COLUMN "intent" varchar(128);--> statement-breakpoint
ALTER TABLE "ai_chatbot"."Chat" ADD COLUMN "customerName" varchar(256);