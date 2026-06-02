<?php

declare(strict_types=1);

namespace App\Http\Controllers\Dashboard;

use App\Extensions\AiCaptions\System\Models\AiCaptionVideo;
use App\Extensions\AiVideoPro\System\Models\UserFall;
use App\Extensions\VideoDubbing\System\Models\VideoDubbing;
use App\Extensions\VideoEditor\System\Models\VideoEditorExportJob;
use App\Helpers\Classes\MarketplaceHelper;
use App\Http\Controllers\Controller;
use App\Models\User;
use Closure;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use Illuminate\View\View;
use Throwable;

class VideoStudioController extends Controller
{
    private const PER_PAGE = 12;

    /**
     * Built-in video tools the dashboard knows about. A tool only appears in the
     * UI when its entry route is registered (i.e. its extension is installed).
     *
     * @return array<int, array{key:string,label:string,description:string,card_image:?string,entry_route:string,resolver:Closure}>
     */
    private function tools(): array
    {
        return array_values(array_filter([
            $this->aiVideoProTool(),
            $this->aiCaptionsTool(),
            $this->videoEditorTool(),
            $this->videoDubbingTool(),
        ]));
    }

    public function __invoke(Request $request): View
    {
        $user = $request->user();
        $tools = $this->tools();

        $todayEntries = collect();
        $previousEntries = collect();
        $paging = [];

        foreach ($tools as $tool) {
            $today = $this->page($tool, $user, 'today', 1);
            $previous = $this->page($tool, $user, 'previous', 1);

            $todayEntries = $todayEntries->concat($today['entries']);
            $previousEntries = $previousEntries->concat($previous['entries']);

            $paging[$tool['key']] = [
                'today'    => ['has_more' => $today['has_more'], 'next_page' => 2],
                'previous' => ['has_more' => $previous['has_more'], 'next_page' => 2],
            ];
        }

        return view('panel.user.video-studio.index', [
            'tools'           => collect($tools),
            'todayEntries'    => $todayEntries->values(),
            'previousEntries' => $previousEntries->values(),
            'paging'          => $paging,
            'perPage'         => self::PER_PAGE,
        ]);
    }

    public function outputs(Request $request): JsonResponse
    {
        $data = $request->validate([
            'source' => 'required|string',
            'bucket' => 'required|string|in:today,previous',
            'page'   => 'required|integer|min:2',
        ]);

        $tool = collect($this->tools())->firstWhere('key', $data['source']);

        if ($tool === null) {
            return response()->json(['error' => __('Unknown source.')], 422);
        }

        $page = (int) $data['page'];
        $result = $this->page($tool, $request->user(), $data['bucket'], $page);

        return response()->json([
            'entries'   => $result['entries'],
            'has_more'  => $result['has_more'],
            'next_page' => $page + 1,
        ]);
    }

    /**
     * @param  array{key:string,resolver:Closure}  $tool
     *
     * @return array{entries: array<int, mixed>, has_more: bool}
     */
    private function page(array $tool, User $user, string $bucket, int $page): array
    {
        try {
            return ($tool['resolver'])($user, $bucket, max(1, $page), self::PER_PAGE);
        } catch (Throwable) {
            return ['entries' => [], 'has_more' => false];
        }
    }

    private function aiVideoProTool(): ?array
    {
        if (! MarketplaceHelper::isRegistered('ai-video-pro') || ! Route::has('dashboard.user.ai-video-pro.index')) {
            return null;
        }

        return [
            'key'         => 'ai-video-pro',
            'label'       => __('AI Video Pro'),
            'description' => __('Frontier AI Generation Models'),
            'card_image'  => asset('upload/images/studio/pro.png'),
            'entry_route' => 'dashboard.user.ai-video-pro.index',
            'resolver'    => fn (User $user, string $bucket, int $page, int $perPage) => $this->paginate(
                UserFall::query()->where('user_id', $user->id),
                $bucket,
                $page,
                $perPage,
                fn (UserFall $v) => [
                    'source'             => 'ai-video-pro',
                    'source_label'       => __('AI Video Pro'),
                    'id'                 => $v->id,
                    'title'              => $v->prompt ? mb_substr((string) $v->prompt, 0, 80) : __('Untitled'),
                    'thumb_url'          => $v->thumbnail_url,
                    'video_url'          => $v->video_url,
                    'status'             => $this->normalizeStatus($v->status, completed: ['complete', 'completed'], failed: ['error', 'failed']),
                    'created_at'         => optional($v->created_at)->diffForHumans(),
                    'status_url'         => Route::has('dashboard.user.ai-video-pro.check')
                        ? route('dashboard.user.ai-video-pro.check') . '?id=' . $v->id
                        : null,
                    'rename_url'         => null,
                    'destroy_url'        => Route::has('dashboard.user.ai-video-pro.delete')
                        ? route('dashboard.user.ai-video-pro.delete', ['id' => $v->id])
                        : null,
                    'prompt'             => $v->prompt,
                    'model'              => $v->model,
                    'resolution'         => $v->resolution,
                    'aspect_ratio'       => $v->aspect_ratio,
                    'formatted_duration' => $v->duration_seconds ? $v->formatted_duration : null,
                ],
            ),
        ];
    }

    private function aiCaptionsTool(): ?array
    {
        if (! MarketplaceHelper::isRegistered('ai-captions') || ! Route::has('dashboard.user.ai-captions.index')) {
            return null;
        }

        return [
            'key'         => 'ai-captions',
            'label'       => __('AI Captions'),
            'description' => __('Add Subtitles to Short Videos'),
            'card_image'  => asset('upload/images/studio/captions.png'),
            'entry_route' => 'dashboard.user.ai-captions.index',
            'resolver'    => fn (User $user, string $bucket, int $page, int $perPage) => $this->paginate(
                AiCaptionVideo::query()->where('user_id', $user->id),
                $bucket,
                $page,
                $perPage,
                fn (AiCaptionVideo $v) => [
                    'source'             => 'ai-captions',
                    'source_label'       => __('AI Captions'),
                    'id'                 => $v->id,
                    'title'              => $v->title ?: __('Untitled'),
                    'thumb_url'          => null,
                    'video_url'          => $v->output_url,
                    'status'             => $this->normalizeStatus(
                        $v->status,
                        completed: ['completed'],
                        failed: ['failed', 'cancelled'],
                    ),
                    'created_at'         => optional($v->created_at)->diffForHumans(),
                    'status_url'         => Route::has('dashboard.user.ai-captions.check')
                        ? route('dashboard.user.ai-captions.check') . '?id=' . $v->id
                        : null,
                    'rename_url'         => null,
                    'destroy_url'        => Route::has('dashboard.user.ai-captions.delete')
                        ? route('dashboard.user.ai-captions.delete', ['id' => $v->id])
                        : null,
                    'template_name'      => $v->template_name,
                    'formatted_duration' => $v->duration_seconds ? $v->formatted_duration : null,
                ],
            ),
        ];
    }

    private function videoEditorTool(): ?array
    {
        if (! MarketplaceHelper::isRegistered('video-editor') || ! Route::has('dashboard.user.video-editor.index')) {
            return null;
        }

        return [
            'key'         => 'video-editor',
            'label'       => __('Video Editor'),
            'description' => __('Professional Video Edit'),
            'card_image'  => asset('upload/images/studio/editor.png'),
            'entry_route' => 'dashboard.user.video-editor.index',
            'resolver'    => fn (User $user, string $bucket, int $page, int $perPage) => $this->paginate(
                VideoEditorExportJob::query()
                    ->with('project')
                    ->where('user_id', $user->id)
                    ->where('status', 'completed')
                    ->whereNotNull('output_path'),
                $bucket,
                $page,
                $perPage,
                function (VideoEditorExportJob $job) {
                    $project = $job->project;
                    $output = $job->output_path;
                    $videoUrl = $output
                        ? (str_starts_with($output, 'http') ? $output : url('uploads/' . $output))
                        : null;
                    $projectName = $project?->name ?: __('Untitled Project');

                    $resolution = $project?->width && $project?->height
                        ? $project->width . '×' . $project->height
                        : null;

                    return [
                        'source'       => 'video-editor',
                        'source_label' => __('Video Editor'),
                        'id'           => $job->id,
                        'title'        => $projectName . ' — ' . __('Export #:id', ['id' => $job->id]),
                        'thumb_url'    => $project?->thumbnail_url,
                        'video_url'    => $videoUrl,
                        'status'       => $this->normalizeStatus(
                            $job->status,
                            completed: ['completed', 'complete', 'finished'],
                            failed: ['failed', 'error', 'cancelled'],
                        ),
                        'created_at'   => optional($job->created_at)->diffForHumans(),
                        'status_url'   => Route::has('dashboard.user.video-editor.export.status')
                            ? route('dashboard.user.video-editor.export.status', ['exportJob' => $job->id])
                            : null,
                        'rename_url'   => Route::has('dashboard.user.video-editor.export.rename')
                            ? route('dashboard.user.video-editor.export.rename', ['exportJob' => $job->id])
                            : null,
                        'destroy_url'  => Route::has('dashboard.user.video-editor.export.delete')
                            ? route('dashboard.user.video-editor.export.delete', ['exportJob' => $job->id])
                            : null,
                        'project_name' => $projectName,
                        'width'        => $project?->width,
                        'height'       => $project?->height,
                        'resolution'   => $resolution,
                    ];
                },
            ),
        ];
    }

    private function videoDubbingTool(): ?array
    {
        if (! MarketplaceHelper::isRegistered('video-dubbing') || ! Route::has('dashboard.user.video-dubbing.index')) {
            return null;
        }

        return [
            'key'         => 'video-dubbing',
            'label'       => __('Video Dubbing'),
            'description' => __('Convert Video To Different Languages'),
            'card_image'  => asset('upload/images/studio/dupping.png'),
            'entry_route' => 'dashboard.user.video-dubbing.index',
            'resolver'    => fn (User $user, string $bucket, int $page, int $perPage) => $this->paginate(
                VideoDubbing::query()->where('user_id', $user->id),
                $bucket,
                $page,
                $perPage,
                fn (VideoDubbing $v) => [
                    'source'             => 'video-dubbing',
                    'source_label'       => __('Video Dubbing'),
                    'id'                 => $v->id,
                    'title'              => $v->title ?: __('Untitled'),
                    'thumb_url'          => null,
                    'video_url'          => $v->output_url,
                    'status'             => $this->normalizeStatus(
                        $v->status,
                        completed: ['complete', 'completed'],
                        failed: ['error', 'failed'],
                    ),
                    'created_at'         => optional($v->created_at)->diffForHumans(),
                    'status_url'         => Route::has('dashboard.user.video-dubbing.check')
                        ? route('dashboard.user.video-dubbing.check') . '?id=' . $v->id
                        : null,
                    'rename_url'         => null,
                    'destroy_url'        => Route::has('dashboard.user.video-dubbing.delete')
                        ? route('dashboard.user.video-dubbing.delete', ['id' => $v->id])
                        : null,
                    'source_language'    => $v->source_language,
                    'target_language'    => $v->target_language,
                    'formatted_duration' => $v->duration_seconds
                        ? sprintf('%02d:%02d', intdiv((int) $v->duration_seconds, 60), (int) $v->duration_seconds % 60)
                        : null,
                ],
            ),
        ];
    }

    /**
     * @return array{entries: array<int, mixed>, has_more: bool}
     */
    private function paginate(Builder $query, string $bucket, int $page, int $perPage, Closure $mapper): array
    {
        $query = $query
            ->when(
                $bucket === 'today',
                fn (Builder $q) => $q->whereDate('created_at', '>=', today()),
                fn (Builder $q) => $q->whereDate('created_at', '<', today()),
            )
            ->latest();

        $total = (clone $query)->count();
        $rows = $query->forPage($page, $perPage)->get();

        return [
            'entries'  => $rows->map($mapper)->all(),
            'has_more' => ($page * $perPage) < $total,
        ];
    }

    /**
     * @param  array<int, string>  $completed
     * @param  array<int, string>  $failed
     */
    private function normalizeStatus(?string $status, array $completed, array $failed): string
    {
        $status = (string) $status;

        if (in_array($status, $completed, true)) {
            return 'completed';
        }

        if (in_array($status, $failed, true)) {
            return 'failed';
        }

        return $status !== '' ? $status : 'pending';
    }
}
