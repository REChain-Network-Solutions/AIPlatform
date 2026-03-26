<?php

namespace Modules\Inspection\Support\Dto;

final class RecurringScheduleDto
{
    public function __construct(
        public readonly int $id,
        public readonly ?string $title = null,
    ) {}
}
