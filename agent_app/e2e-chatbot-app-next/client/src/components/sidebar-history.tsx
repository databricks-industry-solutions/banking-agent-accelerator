import { isToday, isYesterday, subMonths, subWeeks } from 'date-fns';
import { useParams, useNavigate } from 'react-router-dom';
import { useCallback, useState } from 'react';

type ClientUser = {
  email: string;
  name?: string;
  preferredUsername?: string;
};
import { toast } from 'sonner';
import { motion } from 'framer-motion';
import { useConfig } from '@/hooks/use-config';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  useSidebar,
} from '@/components/ui/sidebar';
import type { Chat } from '@chat-template/db';
import { fetcher } from '@/lib/utils';
import { ChatItem } from './sidebar-history-item';
import useSWRInfinite from 'swr/infinite';
import { LoaderIcon, SearchIcon, XIcon } from 'lucide-react';

type GroupedChats = {
  today: Chat[];
  yesterday: Chat[];
  lastWeek: Chat[];
  lastMonth: Chat[];
  older: Chat[];
};

export interface ChatHistory {
  chats: Array<Chat>;
  hasMore: boolean;
}

const PAGE_SIZE = 20;

type StatusFilter = 'open' | 'waiting' | 'received' | 'all';

const INTENT_OPTIONS = [
  { value: '', label: 'All intents' },
  { value: 'ADD_BENEFICIARY', label: 'Add Beneficiary' },
  { value: 'REQUEST_CREDIT_LIMIT_INCREASE', label: 'Credit Limit Increase' },
];

interface FilterState {
  status: StatusFilter;
  intent: string;
  customer: string;
}

const DEFAULT_FILTERS: FilterState = {
  status: 'all',
  intent: '',
  customer: '',
};

function buildFilterQuery(filters: FilterState): string {
  const params = new URLSearchParams();
  if (filters.status !== 'all') params.set('status', filters.status);
  if (filters.intent) params.set('intent', filters.intent);
  if (filters.customer) params.set('customer', filters.customer);
  const qs = params.toString();
  return qs ? `&${qs}` : '';
}

const groupChatsByDate = (chats: Chat[]): GroupedChats => {
  const now = new Date();
  const oneWeekAgo = subWeeks(now, 1);
  const oneMonthAgo = subMonths(now, 1);

  return chats.reduce(
    (groups, chat) => {
      const chatDate = new Date(chat.createdAt);

      if (isToday(chatDate)) {
        groups.today.push(chat);
      } else if (isYesterday(chatDate)) {
        groups.yesterday.push(chat);
      } else if (chatDate > oneWeekAgo) {
        groups.lastWeek.push(chat);
      } else if (chatDate > oneMonthAgo) {
        groups.lastMonth.push(chat);
      } else {
        groups.older.push(chat);
      }

      return groups;
    },
    {
      today: [],
      yesterday: [],
      lastWeek: [],
      lastMonth: [],
      older: [],
    } as GroupedChats,
  );
};

export function getChatHistoryPaginationKey(
  pageIndex: number,
  previousPageData: ChatHistory,
  filterQuery = '',
) {
  if (previousPageData && previousPageData.hasMore === false) {
    return null;
  }

  if (pageIndex === 0)
    return `/api/history?limit=${PAGE_SIZE}${filterQuery}`;

  const firstChatFromPage = previousPageData.chats.at(-1);

  if (!firstChatFromPage) return null;

  return `/api/history?ending_before=${firstChatFromPage.id}&limit=${PAGE_SIZE}${filterQuery}`;
}

function ChatDateGroup({
  label,
  chats,
  activeId,
  onDelete,
  setOpenMobile,
}: {
  label: string;
  chats: Chat[];
  activeId?: string;
  onDelete: (id: string) => void;
  setOpenMobile: (open: boolean) => void;
}) {
  if (chats.length === 0) return null;
  return (
    <div>
      <div className="px-2 py-1 text-sidebar-foreground/50 text-xs">
        {label}
      </div>
      {chats.map((chat) => (
        <ChatItem
          key={chat.id}
          chat={chat}
          isActive={chat.id === activeId}
          onDelete={onDelete}
          setOpenMobile={setOpenMobile}
        />
      ))}
    </div>
  );
}

function FilterBar({
  filters,
  onChange,
}: {
  filters: FilterState;
  onChange: (filters: FilterState) => void;
}) {
  const hasActiveFilters =
    filters.status !== 'all' || filters.intent !== '' || filters.customer !== '';

  return (
    <div className="flex flex-col gap-1.5 px-2 pb-2">
      <div className="flex items-center gap-1">
        {(['open', 'waiting', 'received'] as const).map((value) => (
          <button
            key={value}
            type="button"
            onClick={() =>
              onChange({
                ...filters,
                status: filters.status === value ? 'all' : value,
              })
            }
            className={`rounded-md px-2 py-0.5 text-xs font-medium transition-colors ${
              filters.status === value
                ? 'bg-primary text-primary-foreground'
                : 'bg-sidebar-accent text-sidebar-accent-foreground hover:bg-sidebar-accent/80'
            }`}
          >
            {{ open: 'Open', waiting: 'Waiting', received: 'Received' }[value]}
          </button>
        ))}
        <select
          value={filters.intent}
          onChange={(e) => onChange({ ...filters, intent: e.target.value })}
          className="h-6 flex-1 rounded-md border-0 bg-sidebar-accent px-1.5 text-xs text-sidebar-accent-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        >
          {INTENT_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        {hasActiveFilters && (
          <button
            type="button"
            onClick={() => onChange(DEFAULT_FILTERS)}
            className="flex h-5 w-5 items-center justify-center rounded-md text-sidebar-foreground/50 hover:bg-sidebar-accent hover:text-sidebar-foreground"
          >
            <XIcon size={12} />
          </button>
        )}
      </div>
      <div className="relative">
        <SearchIcon
          size={12}
          className="absolute left-2 top-1/2 -translate-y-1/2 text-sidebar-foreground/40"
        />
        <input
          type="text"
          value={filters.customer}
          onChange={(e) => onChange({ ...filters, customer: e.target.value })}
          placeholder="Search customer..."
          className="h-6 w-full rounded-md border-0 bg-sidebar-accent pl-6 pr-2 text-xs text-sidebar-accent-foreground placeholder:text-sidebar-foreground/40 focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>
    </div>
  );
}

export function SidebarHistory({ user }: { user?: ClientUser | null }) {
  const { setOpenMobile } = useSidebar();
  const { id } = useParams();
  const { chatHistoryEnabled } = useConfig();
  const [filters, setFilters] = useState<FilterState>(DEFAULT_FILTERS);

  const filterQuery = buildFilterQuery(filters);

  const paginationKeyFn = useCallback(
    (pageIndex: number, previousPageData: ChatHistory) =>
      getChatHistoryPaginationKey(pageIndex, previousPageData, filterQuery),
    [filterQuery],
  );

  const {
    data: paginatedChatHistories,
    setSize,
    isValidating,
    isLoading,
    mutate,
  } = useSWRInfinite<ChatHistory>(paginationKeyFn, fetcher, {
    fallbackData: [],
  });

  const navigate = useNavigate();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const hasReachedEnd = paginatedChatHistories
    ? paginatedChatHistories.some((page) => page.hasMore === false)
    : false;

  const hasEmptyChatHistory = paginatedChatHistories
    ? paginatedChatHistories.every((page) => page.chats.length === 0)
    : false;

  const handleDelete = async () => {
    const deletePromise = fetch(`/api/chat/${deleteId}`, {
      method: 'DELETE',
    });

    toast.promise(deletePromise, {
      loading: 'Deleting chat...',
      success: () => {
        mutate((chatHistories) => {
          if (chatHistories) {
            return chatHistories.map((chatHistory) => ({
              ...chatHistory,
              chats: chatHistory.chats.filter((chat) => chat.id !== deleteId),
            }));
          }
        });

        return 'Chat deleted successfully';
      },
      error: 'Failed to delete chat',
    });

    setShowDeleteDialog(false);

    if (window.location.pathname === `/chat/${deleteId}`) {
      navigate('/');
    }

    if (deleteId === id) {
      navigate('/');
    }
  };

  const onDeleteChat = useCallback((chatId: string) => {
    setDeleteId(chatId);
    setShowDeleteDialog(true);
  }, []);

  if (!user) {
    return (
      <SidebarGroup>
        <SidebarGroupContent>
          <div className="flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-zinc-500">
            Login to save and revisit previous chats!
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  if (isLoading) {
    return (
      <SidebarGroup>
        <div className="px-2 py-1 text-sidebar-foreground/50 text-xs">
          Today
        </div>
        <SidebarGroupContent>
          <div className="flex flex-col">
            {[44, 32, 28, 64, 52].map((item) => (
              <div
                key={item}
                className="flex h-8 items-center gap-2 rounded-md px-2"
              >
                <div
                  className="h-4 max-w-(--skeleton-width) flex-1 rounded-md bg-sidebar-accent-foreground/10"
                  style={
                    {
                      '--skeleton-width': `${item}%`,
                    } as React.CSSProperties
                  }
                />
              </div>
            ))}
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  return (
    <>
      <SidebarGroup>
        <SidebarGroupContent>
          <FilterBar filters={filters} onChange={setFilters} />

          {hasEmptyChatHistory ? (
            <div className="flex w-full flex-row items-center justify-center gap-2 px-2 py-4 text-sm text-zinc-500">
              {chatHistoryEnabled
                ? filters.status !== 'all' ||
                    filters.intent ||
                    filters.customer
                  ? 'No chats match your filters.'
                  : 'Your conversations will appear here once you start chatting!'
                : 'Chat history is disabled - conversations are not saved'}
            </div>
          ) : (
            <>
              <SidebarMenu>
                {paginatedChatHistories &&
                  (() => {
                    const chatsFromHistory = paginatedChatHistories.flatMap(
                      (paginatedChatHistory) => paginatedChatHistory.chats,
                    );

                    const groupedChats = groupChatsByDate(chatsFromHistory);

                    return (
                      <div className="flex flex-col gap-6">
                        <ChatDateGroup
                          label="Today"
                          chats={groupedChats.today}
                          activeId={id}
                          onDelete={onDeleteChat}
                          setOpenMobile={setOpenMobile}
                        />
                        <ChatDateGroup
                          label="Yesterday"
                          chats={groupedChats.yesterday}
                          activeId={id}
                          onDelete={onDeleteChat}
                          setOpenMobile={setOpenMobile}
                        />
                        <ChatDateGroup
                          label="Last 7 days"
                          chats={groupedChats.lastWeek}
                          activeId={id}
                          onDelete={onDeleteChat}
                          setOpenMobile={setOpenMobile}
                        />
                        <ChatDateGroup
                          label="Last 30 days"
                          chats={groupedChats.lastMonth}
                          activeId={id}
                          onDelete={onDeleteChat}
                          setOpenMobile={setOpenMobile}
                        />
                        <ChatDateGroup
                          label="Older than last month"
                          chats={groupedChats.older}
                          activeId={id}
                          onDelete={onDeleteChat}
                          setOpenMobile={setOpenMobile}
                        />
                      </div>
                    );
                  })()}
              </SidebarMenu>

              <motion.div
                onViewportEnter={() => {
                  if (!isValidating && !hasReachedEnd) {
                    setSize((size) => size + 1);
                  }
                }}
              />

              {hasReachedEnd ? (
                <div className="mt-8 flex w-full flex-row items-center justify-center gap-2 px-2 text-sm text-zinc-500">
                  You have reached the end of your chat history.
                </div>
              ) : (
                <div className="mt-8 flex flex-row items-center gap-2 p-2 text-zinc-500 dark:text-zinc-400">
                  <div className="animate-spin">
                    <LoaderIcon />
                  </div>
                  <div>Loading Chats...</div>
                </div>
              )}
            </>
          )}
        </SidebarGroupContent>
      </SidebarGroup>

      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete your
              chat and remove it from our servers.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete}>
              Continue
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
