<?php

namespace Modules\QualityControl\Support\Dto;

final class InspectionSummaryDto
{
    public function __construct(
        public readonly int $id,
        public readonly string $status,
    ) {}
}
