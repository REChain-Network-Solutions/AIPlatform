<?php

namespace App\View\Components\Forms;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class SelectboxOption extends Component
{
    public function __construct(
        public string $value,
        public string $label,
    ) {}

    public function render(): View|Closure|string
    {
        return view('components.forms.selectbox-option');
    }
}
