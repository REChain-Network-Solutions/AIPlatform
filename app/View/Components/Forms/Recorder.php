<?php

namespace App\View\Components\Forms;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Recorder extends Component
{
    public function __construct(
        public ?string $name = null,
        public ?string $label = null,
        public ?string $placeholder = null,
        public ?string $tooltip = null,
        public ?int $maxDurationSeconds = null,
        public ?int $maxSizeMb = null,
    ) {}

    public function render(): View|Closure|string
    {
        return view('components.forms.recorder');
    }
}
