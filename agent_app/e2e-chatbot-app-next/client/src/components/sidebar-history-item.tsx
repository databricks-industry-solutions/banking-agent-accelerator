import type { Chat } from '@chat-template/db';
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from './ui/sidebar';
import { Link } from 'react-router-dom';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuPortal,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import { memo } from 'react';
import { useChatVisibility } from '@/hooks/use-chat-visibility';
import {
  CircleCheck,
  GlobeIcon,
  LockIcon,
  MoreHorizontalIcon,
  ShareIcon,
  TrashIcon,
} from 'lucide-react';

const STAGE_STEP: Record<string, number> = {
  START: 1,
  CLASSIFY_INTENT: 1,
  GET_TEMPLATE: 2,
  ASK_FOR_FIELDS: 3,
  EXTRACT_FIELDS: 3,
  LOOKUP_CUSTOMER_EMAIL: 4,
  CUSTOMER_BACKGROUND_CHECK: 5,
  WAITING_FOR_BACKGROUND_CHECK: 5,
  BACKGROUND_CHECK_RECEIVED: 5,
  DENIED: 5,
  ERROR: 0,
  CONFIRM: 6,
  SEND_EMAIL: 7,
  DONE: 8,
};

const TOTAL_STEPS = 8;

const INTENT_LABELS: Record<string, string> = {
  GENERATE_ACCOUNT_STATEMENT: 'Statement',
  OPEN_DEPOSIT: 'Deposit',
};

function WorkflowBadges({ chat }: { chat: Chat }) {
  const stage = chat.stage;
  if (!stage) return null;

  const step = STAGE_STEP[stage] ?? 0;
  const isDone = stage === 'DONE';
  const isWaiting = stage === 'WAITING_FOR_BACKGROUND_CHECK';
  const isReceived = stage === 'BACKGROUND_CHECK_RECEIVED';
  const isDenied = stage === 'DENIED';
  const isError = stage === 'ERROR';
  const intentLabel = chat.intent ? INTENT_LABELS[chat.intent] : null;

  const statusBadge = isWaiting ? (
    <span className="animate-pulse rounded bg-amber-500/15 px-1 py-px text-[10px] font-medium leading-tight text-amber-600 dark:text-amber-400">
      Waiting…
    </span>
  ) : isReceived ? (
    <span className="rounded bg-green-500/15 px-1 py-px text-[10px] font-medium leading-tight text-green-600 dark:text-green-400">
      Answer received
    </span>
  ) : isDenied ? (
    <span className="rounded bg-red-500/15 px-1 py-px text-[10px] font-medium leading-tight text-red-600 dark:text-red-400">
      Denied
    </span>
  ) : isError ? (
    <span className="rounded bg-red-500/15 px-1 py-px text-[10px] font-medium leading-tight text-red-600 dark:text-red-400">
      Error
    </span>
  ) : (
    <span
      className={`rounded px-1 py-px text-[10px] font-medium leading-tight ${
        isDone
          ? 'bg-green-500/15 text-green-600 dark:text-green-400'
          : 'bg-amber-500/15 text-amber-600 dark:text-amber-400'
      }`}
    >
      {isDone ? 'Done' : `${step}/${TOTAL_STEPS}`}
    </span>
  );

  return (
    <div className="flex items-center gap-1 px-1 pt-0.5">
      {intentLabel && (
        <span className="rounded bg-blue-500/15 px-1 py-px text-[10px] font-medium leading-tight text-blue-600 dark:text-blue-400">
          {intentLabel}
        </span>
      )}
      {statusBadge}
      {chat.customerName && (
        <span className="truncate text-[10px] leading-tight text-sidebar-foreground/50">
          {chat.customerName}
        </span>
      )}
    </div>
  );
}

const PureChatItem = ({
  chat,
  isActive,
  onDelete,
  setOpenMobile,
}: {
  chat: Chat;
  isActive: boolean;
  onDelete: (chatId: string) => void;
  setOpenMobile: (open: boolean) => void;
}) => {
  const { visibilityType, setVisibilityType } = useChatVisibility({
    chatId: chat.id,
    initialVisibilityType: chat.visibility,
  });

  return (
    <SidebarMenuItem data-testid="chat-history-item">
      <SidebarMenuButton asChild isActive={isActive}>
        <Link to={`/chat/${chat.id}`} onClick={() => setOpenMobile(false)}>
          <div className="flex min-w-0 flex-col">
            <span className="truncate">{chat.title}</span>
            <WorkflowBadges chat={chat} />
          </div>
        </Link>
      </SidebarMenuButton>

      <DropdownMenu modal={true}>
        <DropdownMenuTrigger asChild>
          <SidebarMenuAction
            data-testid="chat-options"
            className="mr-0.5 data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            showOnHover={!isActive}
          >
            <MoreHorizontalIcon />
            <span className="sr-only">More</span>
          </SidebarMenuAction>
        </DropdownMenuTrigger>

        <DropdownMenuContent side="bottom" align="end">
          <DropdownMenuSub>
            <DropdownMenuSubTrigger className="cursor-pointer">
              <ShareIcon />
              <span>Share</span>
            </DropdownMenuSubTrigger>
            <DropdownMenuPortal>
              <DropdownMenuSubContent>
                <DropdownMenuItem
                  className="cursor-pointer flex-row justify-between"
                  onClick={() => {
                    setVisibilityType('private');
                  }}
                >
                  <div className="flex flex-row items-center gap-2">
                    <LockIcon size={12} />
                    <span>Private</span>
                  </div>
                  {visibilityType === 'private' ? <CircleCheck /> : null}
                </DropdownMenuItem>
                <DropdownMenuItem
                  className="cursor-pointer flex-row justify-between"
                  onClick={() => {
                    setVisibilityType('public');
                  }}
                >
                  <div className="flex flex-row items-center gap-2">
                    <GlobeIcon />
                    <span>Public</span>
                  </div>
                  {visibilityType === 'public' ? <CircleCheck /> : null}
                </DropdownMenuItem>
              </DropdownMenuSubContent>
            </DropdownMenuPortal>
          </DropdownMenuSub>

          <DropdownMenuItem
            className="cursor-pointer text-destructive focus:bg-destructive/15 focus:text-destructive dark:text-red-500"
            onSelect={() => onDelete(chat.id)}
          >
            <TrashIcon />
            <span>Delete</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </SidebarMenuItem>
  );
};

export const ChatItem = memo(PureChatItem, (prevProps, nextProps) => {
  if (prevProps.isActive !== nextProps.isActive) return false;
  if (prevProps.chat.stage !== nextProps.chat.stage) return false;
  if (prevProps.chat.intent !== nextProps.chat.intent) return false;
  if (prevProps.chat.customerName !== nextProps.chat.customerName) return false;
  return true;
});
