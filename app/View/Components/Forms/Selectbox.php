<?php

namespace App\View\Components\Forms;

use Closure;
use Illuminate\Contracts\View\View;
use Illuminate\View\Component;

class Selectbox extends Component
{
    public string $placeholder;

    public ?string $selectedValue;

    public ?string $activeLabel;

    public function __construct(
        public ?string $name = null,
        public ?string $label = null,
        public ?string $value = null,
        ?string $placeholder = null,
        public ?string $tooltip = null,
        public array $options = [],
    ) {
        $this->placeholder = $placeholder ?? __('Select an option');
        $this->selectedValue = $value;
        $this->activeLabel = $this->resolveActiveLabel();
    }

    private function resolveActiveLabel(): ?string
    {
        if ($this->value === null || $this->value === '') {
            return null;
        }

        foreach ($this->options as $option) {
            if ((string) ($option['value'] ?? '') === (string) $this->value) {
                return (string) ($option['label'] ?? '');
            }
        }

        return null;
    }

    public function render(): View|Closure|string
    {
        return view('components.forms.selectbox');
    }
}
