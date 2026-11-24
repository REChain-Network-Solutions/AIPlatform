<?php

namespace App\Services\Common;

use App\Helpers\Classes\PlanHelper;
use App\Models\OpenAIGenerator;
use App\Models\Team\Team;
use App\Models\User;
use App\Models\UserOpenai;
use App\Models\UserOpenaiChat;
use Exception;
use Illuminate\Support\Str;

class GlobalSearchService
{
    private const MAX_PER_GROUP = 5;

    public function __construct(private readonly MenuService $menuService) {}

    public function search(User $user, string $query): array
    {
        $groups = [];

        foreach ([
            $this->searchNavigation($user, $query),
            $this->searchSettings($user, $query),
            $this->searchWorkspaces($user, $query),
            $this->searchDocuments($user, $query),
            $this->searchChats($user, $query),
            $this->searchTemplates($user, $query),
        ] as $group) {
            if (! empty($group['items'])) {
                $groups[] = $group;
            }
        }

        return $groups;
    }

    private function searchNavigation(User $user, string $query): array
    {
        $menu = $this->menuService->generate();
        $lowerQuery = strtolower($query);
        $items = [];

        foreach ($menu as $menuItem) {
            if (! $this->isMenuItemVisible($menuItem, $user)) {
                continue;
            }

            if (($menuItem['type'] ?? '') === 'item') {
                if (Str::contains(strtolower($menuItem['label'] ?? ''), $lowerQuery)) {
                    $items[] = $this->formatMenuItem($menuItem);
                }
            }

            foreach ($menuItem['children'] ?? [] as $child) {
                if (! $this->isMenuItemVisible($child, $user) || ($child['type'] ?? '') !== 'item') {
                    continue;
                }

                if (Str::contains(strtolower($child['label'] ?? ''), $lowerQuery)) {
                    $items[] = $this->formatMenuItem($child, $menuItem['label'] ?? null);
                }
            }
        }

        return [
            'type'  => 'navigation',
            'label' => __('Navigation'),
            'items' => array_slice($items, 0, self::MAX_PER_GROUP),
        ];
    }

    private function isMenuItemVisible(array $item, User $user): bool
    {
        if (! ($item['show_condition'] ?? true)) {
            return false;
        }

        if ($item['is_admin'] ?? false) {
            return $user->isAdmin();
        }

        return true;
    }

    private function formatMenuItem(array $item, ?string $context = null): array
    {
        $url = '#';

        if (! empty($item['route'])) {
            try {
                $url = route($item['route']);
            } catch (Exception) {
                // Route may not exist when extension is not registered
            }
        }

        return [
            'svg'     => $this->renderIcon($item['icon'] ?? 'tabler-layout-2'),
            'label'   => $item['label'] ?? '',
            'context' => $context ?? __('Navigation'),
            'url'     => $url,
        ];
    }

    private function renderIcon(string $iconName, string $classes = 'size-4'): string
    {
        try {
            return svg($iconName, ['class' => $classes, 'stroke-width' => '1.5'])->toHtml();
        } catch (Exception) {
            return svg('tabler-layout-2', ['class' => $classes, 'stroke-width' => '1.5'])->toHtml();
        }
    }

    private function searchSettings(User $user, string $query): array
    {
        $lowerQuery = strtolower($query);

        $items = collect($this->getSettingsDefinitions($user))
            ->filter(fn ($s) => Str::contains(strtolower($s['label']), $lowerQuery))
            ->take(self::MAX_PER_GROUP)
            ->values()
            ->toArray();

        return [
            'type'  => 'settings',
            'label' => __('Settings'),
            'items' => $items,
        ];
    }

    private function getSettingsDefinitions(User $user): array
    {
        $userSettings = [
            ['svg' => $this->renderIcon('tabler-settings'), 'label' => __('Profile Settings'), 'context' => __('User Settings'), 'url' => route('dashboard.user.settings.index')],
        ];

        if (! $user->isAdmin()) {
            return $userSettings;
        }

        return array_merge($userSettings, [
            ['svg' => $this->renderIcon('tabler-settings-2'), 'label' => __('General Settings'), 'context' => __('Admin Settings'), 'url' => route('dashboard.admin.settings.general')],
            ['svg' => $this->renderIcon('tabler-robot'), 'label' => __('OpenAI Settings'), 'context' => __('Admin Settings'), 'url' => route('dashboard.admin.settings.openai')],
            ['svg' => $this->renderIcon('tabler-palette'), 'label' => __('Frontend Settings'), 'context' => __('Admin Settings'), 'url' => route('dashboard.admin.frontend.settings')],
            ['svg' => $this->renderIcon('tabler-menu'), 'label' => __('Menu Settings'), 'context' => __('Admin Settings'), 'url' => route('dashboard.admin.frontend.menusettings')],
            ['svg' => $this->renderIcon('tabler-coin'), 'label' => __('Finance & Plans'), 'context' => __('Admin Settings'), 'url' => route('dashboard.admin.finance.plan.index')],
        ]);
    }

    private function searchWorkspaces(User $user, string $query): array
    {
        $items = Team::query()
            ->where(function ($q) use ($user) {
                $q->where('user_id', $user->id)
                    ->orWhereHas('members', fn ($m) => $m->where('user_id', $user->id));
            })
            ->where('name', 'like', "%{$query}%")
            ->select('id', 'name', 'user_id')
            ->limit(self::MAX_PER_GROUP)
            ->get()
            ->map(fn ($team) => [
                'svg'     => $this->renderIcon('tabler-users'),
                'label'   => $team->name,
                'context' => $team->user_id === $user->id ? __('Your workspace') : __('Shared workspace'),
                'url'     => route('dashboard.user.team.index'),
            ])
            ->toArray();

        return [
            'type'  => 'workspaces',
            'label' => __('Workspaces'),
            'items' => $items,
        ];
    }

    private function searchDocuments(User $user, string $query): array
    {
        $items = collect();

        // Core documents (user_openai)
        UserOpenai::query()
            ->where('user_id', $user->id)
            ->where(function ($q) use ($query) {
                $q->where('title', 'like', "%{$query}%")
                    ->orWhere('output', 'like', "%{$query}%");
            })
            ->select('id', 'title', 'slug', 'openai_id', 'output', 'input')
            ->with('generator:id,title,slug,type')
            ->limit(self::MAX_PER_GROUP)
            ->get()
            ->each(function ($doc) use ($items) {
                $items->push([
                    'svg'     => $this->renderIcon($this->documentIcon($doc->generator?->type)),
                    'label'   => $this->documentLabel($doc),
                    'context' => $doc->generator?->title ?? __('Document'),
                    'url'     => route('dashboard.user.openai.documents.single', $doc->slug),
                ]);
            });

        // AI Video Pro (user_fall)
        $userFallClass = 'App\Extensions\AiVideoPro\System\Models\UserFall';
        if (class_exists($userFallClass)) {
            $userFallClass::query()
                ->where('user_id', $user->id)
                ->where('status', 'complete')
                ->whereNotNull('video_url')
                ->where('video_url', '!=', '')
                ->where('prompt', 'like', "%{$query}%")
                ->select('id', 'prompt')
                ->limit(self::MAX_PER_GROUP)
                ->get()
                ->each(function ($video) use ($items) {
                    $words = explode(' ', trim($video->prompt ?? ''));
                    $title = implode(' ', array_slice($words, 0, 6)) . (count($words) > 6 ? '...' : '');
                    $items->push([
                        'svg'     => $this->renderIcon('tabler-video'),
                        'label'   => $title ?: __('Untitled Video'),
                        'context' => __('AI Video'),
                        'url'     => route('dashboard.user.openai.documents.single', Str::slug($title)),
                    ]);
                });
        }

        // AI Image Pro (ai_image_pro)
        $imageProClass = 'App\Extensions\AIImagePro\System\Models\AiImageProModel';
        if (class_exists($imageProClass)) {
            $imageProClass::query()
                ->where('user_id', $user->id)
                ->where('prompt', 'like', "%{$query}%")
                ->select('id', 'prompt')
                ->limit(self::MAX_PER_GROUP)
                ->get()
                ->each(function ($img) use ($items) {
                    $items->push([
                        'svg'     => $this->renderIcon('tabler-photo'),
                        'label'   => Str::limit($img->prompt ?? __('Untitled Image'), 80),
                        'context' => __('AI Image'),
                        'url'     => route('dashboard.user.openai.documents.single', "ai-image-pro-{$img->id}-0"),
                    ]);
                });
        }

        // AI Chat Pro Image Chat (ai_chat_pro_image)
        $chatImageClass = 'App\Extensions\AiChatProImageChat\System\Models\AiChatProImageModel';
        if (class_exists($chatImageClass)) {
            $chatImageClass::query()
                ->where('user_id', $user->id)
                ->where('prompt', 'like', "%{$query}%")
                ->select('id', 'prompt')
                ->limit(self::MAX_PER_GROUP)
                ->get()
                ->each(function ($img) use ($items) {
                    $items->push([
                        'svg'     => $this->renderIcon('tabler-photo'),
                        'label'   => Str::limit($img->prompt ?? __('Untitled Image'), 80),
                        'context' => __('AI Chat Image'),
                        'url'     => route('dashboard.user.openai.documents.single', "ai-chat-pro-image-chat-{$img->id}-0"),
                    ]);
                });
        }

        // AI Music (user_music)
        $musicClass = 'App\Extensions\AiMusic\System\Models\UserMusic';
        if (class_exists($musicClass)) {
            $musicClass::query()
                ->where('user_id', $user->id)
                ->where('title', 'like', "%{$query}%")
                ->select('id', 'title')
                ->limit(self::MAX_PER_GROUP)
                ->get()
                ->each(function ($music) use ($items) {
                    $items->push([
                        'svg'     => $this->renderIcon('tabler-music'),
                        'label'   => $music->title ?: __('Untitled Music'),
                        'context' => __('AI Music'),
                        'url'     => route('dashboard.user.ai-music.index'),
                    ]);
                });
        }

        // AI Music Pro (ai_music_pro)
        $musicProClass = 'App\Extensions\AiMusicPro\System\Models\AiMusicPro';
        if (class_exists($musicProClass)) {
            $musicProClass::query()
                ->where('user_id', $user->id)
                ->where(function ($q) use ($query) {
                    $q->where('workbook_title', 'like', "%{$query}%")
                        ->orWhere('ai_music_prompt', 'like', "%{$query}%");
                })
                ->select('id', 'workbook_title', 'ai_music_prompt')
                ->limit(self::MAX_PER_GROUP)
                ->get()
                ->each(function ($music) use ($items) {
                    $items->push([
                        'svg'     => $this->renderIcon('tabler-music'),
                        'label'   => $music->workbook_title ?: Str::limit($music->ai_music_prompt ?? __('Untitled Music'), 80),
                        'context' => __('AI Music Pro'),
                        'url'     => route('dashboard.user.ai-music-pro.index'),
                    ]);
                });
        }

        return [
            'type'  => 'documents',
            'label' => __('Documents'),
            'items' => $items->take(self::MAX_PER_GROUP * 2)->values()->toArray(),
        ];
    }

    private function documentLabel(UserOpenai $doc): string
    {
        $type = $doc->generator?->type;
        $limit = 80;

        if (in_array($type, ['text', 'youtube', 'rss', 'code'])) {
            $raw = $doc->title ? $doc->title . ' : ' . ($doc->output ?? '') : ($doc->output ?? '');

            return Str::limit(strip_tags($raw), $limit);
        }

        if (in_array($type, ['image', 'video'])) {
            return Str::limit(strip_tags($doc->input ?? $doc->title ?? ''), $limit);
        }

        if (in_array($type, ['audio', 'voiceover', 'isolator'])) {
            return Str::limit(strip_tags($doc->title ?? ''), $limit);
        }

        return $doc->title ?: __('Untitled');
    }

    private function documentIcon(?string $type): string
    {
        return match ($type) {
            'image' => 'tabler-photo',
            'video' => 'tabler-video',
            'audio', 'voiceover', 'isolator' => 'tabler-microphone',
            'code' => 'tabler-code',
            default => 'tabler-file-text',
        };
    }

    private function searchChats(User $user, string $query): array
    {
        $items = UserOpenaiChat::query()
            ->where('user_id', $user->id)
            ->where('title', 'like', "%{$query}%")
            ->select('id', 'title')
            ->limit(self::MAX_PER_GROUP)
            ->get()
            ->map(fn ($chat) => [
                'svg'     => $this->renderIcon('tabler-message'),
                'label'   => $chat->title ?: __('Untitled chat'),
                'context' => __('AI Chat'),
                'url'     => route('dashboard.user.openai.chat.chat'),
            ])
            ->toArray();

        return [
            'type'  => 'chats',
            'label' => __('Chats'),
            'items' => $items,
        ];
    }

    private function searchTemplates(User $user, string $query): array
    {
        $plan = $user->isAdmin() ? null : $user->relationPlan;

        $items = OpenAIGenerator::query()
            ->where('active', 1)
            ->where('title', 'like', "%{$query}%")
            ->select('id', 'title', 'slug', 'premium')
            ->limit(self::MAX_PER_GROUP * 2)
            ->get()
            ->filter(function ($template) use ($user, $plan) {
                if ($user->isAdmin()) {
                    return true;
                }

                if (! $template->premium) {
                    return true;
                }

                return $plan && PlanHelper::planMenuCheck($plan, $template->slug);
            })
            ->take(self::MAX_PER_GROUP)
            ->map(fn ($template) => [
                'svg'     => $this->renderIcon('tabler-template'),
                'label'   => $template->title,
                'context' => __('Template'),
                'url'     => route('dashboard.user.openai.generator', $template->slug),
            ])
            ->values()
            ->toArray();

        return [
            'type'  => 'templates',
            'label' => __('Templates'),
            'items' => $items,
        ];
    }
}
