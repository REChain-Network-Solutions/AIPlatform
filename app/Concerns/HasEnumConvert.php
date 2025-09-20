<?php

namespace App\Concerns;

use Illuminate\Support\Arr;

trait HasEnumConvert
{
    use HasLog;

    /**
     * get enum from string
     */
    public static function fromValue(string $value): self
    {
        $self = self::tryFrom($value);

        if ($self === null) {
            self::InvalidTypeLog('fromValue', $value);
        }

        return $self;
    }

    /**
     * get Enum from label
     */
    public static function fromLabel(string $label): self
    {
        $match = Arr::first(self::cases(), static fn ($enum) => $enum->label() == $label);

        if (! $match) {
            self::InvalidTypeLog('fromLabel', $label);
        }

        return $match;
    }

    /**
     * get labels from enum
     */
    public static function getLabels(): array
    {
        return Arr::map(self::cases(), static fn ($enum) => $enum->label());
    }

    /**
     * get values
     */
    public static function getValues(): array
    {
        return Arr::map(self::cases(), static fn ($enum) => $enum->value);
    }

    /**
     * get label from enum
     */
    public function label(): string
    {
        return $this->value;
    }
}
