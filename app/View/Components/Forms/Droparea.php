<?php

namespace App\View\Components\Forms;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Droparea extends Component
{
    public function __construct(
        public ?string $style = 'compact',
        public ?string $name = null,
        public ?string $label = null,
        public ?string $accept = null,
        public ?string $placeholder = null,
        public ?string $tooltip = null,
        public ?string $decorator = null,
        public ?array $terms = null,
        public ?int $maxSizeMb = null,
        public bool $multiple = false,
    ) {}

    public function render(): View|Closure|string
    {
        return view('components.forms.droparea');
    }
}
