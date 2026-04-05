<?php
namespace App\Services\CustomModules\Support;

class InstallHistoryFormatter
{
    public function format(array $events): array
    {
        return array_map(fn ($event) => [
            'label' => ucfirst(str_replace('-', ' ', $event['event'] ?? 'event')),
            'time' => $event['time'] ?? null,
        ], $events);
    }
}
