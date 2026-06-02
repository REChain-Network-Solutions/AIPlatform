<?php

declare(strict_types=1);

namespace App\Services\UGCStudio;

use App\Models\User;
use Closure;
use Illuminate\Support\Collection;
use Throwable;

/**
 * Describes a UGC parent extension that contributes entries to the UGC Studio.
 *
 * Each parent extension (UGC Factory, UGC Marketing, UGC Creator, etc.) registers
 * an instance of this DTO with UGCSourceRegistry from its own ServiceProvider boot().
 */
class UGCSource
{
    /**
     * @param  string  $key  Unique identifier (e.g. "ugc-factory").
     * @param  string  $label  human-readable label shown on the entry card
     * @param  string  $icon  Tabler icon name (e.g. "wand") or asset path.
     * @param  string  $entryRoute  route name the entry card links to
     * @param  Closure(User): Collection|null  $outputsResolver  legacy resolver returning the user's outputs un-paged + un-bucketed
     * @param  Closure(User, string, int, int): array|null  $pagedOutputsResolver  paged resolver: (user, bucket, page, perPage) => ['entries' => array, 'has_more' => bool]
     */
    public function __construct(
        public readonly string $key,
        public readonly string $label,
        public readonly string $icon,
        public readonly string $entryRoute,
        public readonly ?Closure $outputsResolver = null,
        public readonly ?string $description = null,
        public readonly ?string $cardImage = null,
        public readonly ?Closure $pagedOutputsResolver = null,
    ) {}

    public function outputs(User $user): Collection
    {
        if ($this->outputsResolver !== null) {
            try {
                return ($this->outputsResolver)($user);
            } catch (Throwable) {
                return collect();
            }
        }

        if ($this->pagedOutputsResolver === null) {
            return collect();
        }

        // Synthesize from the paged resolver for any consumer still using the
        // legacy un-bucketed API — pull a generous first page from each bucket.
        try {
            $today = collect(($this->pagedOutputsResolver)($user, 'today', 1, 100)['entries'] ?? []);
            $previous = collect(($this->pagedOutputsResolver)($user, 'previous', 1, 100)['entries'] ?? []);

            return $today->concat($previous);
        } catch (Throwable) {
            return collect();
        }
    }

    /**
     * One page of entries in the requested bucket.
     *
     * @return array{entries: array<int, mixed>, has_more: bool}
     */
    public function pagedOutputs(User $user, string $bucket, int $page, int $perPage = 12): array
    {
        $page = max(1, $page);
        $perPage = max(1, $perPage);

        if ($this->pagedOutputsResolver !== null) {
            try {
                $result = ($this->pagedOutputsResolver)($user, $bucket, $page, $perPage);
            } catch (Throwable) {
                return ['entries' => [], 'has_more' => false];
            }

            return [
                'entries'  => array_values($result['entries'] ?? []),
                'has_more' => (bool) ($result['has_more'] ?? false),
            ];
        }

        // Legacy fallback: emit everything on page 1 of "today", nothing afterwards.
        if ($page > 1 || $bucket !== 'today') {
            return ['entries' => [], 'has_more' => false];
        }

        return [
            'entries'  => $this->outputs($user)->all(),
            'has_more' => false,
        ];
    }
}
