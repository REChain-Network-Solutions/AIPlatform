<?php
namespace App\Services\CustomModules\DTO;

class ModuleFileRecord
{
    public function __construct(
        public string $name,
        public string $path,
        public int $size,
        public int $modified,
    ) {}

    public function toArray(): array
    {
        return ['name' => $this->name, 'path' => $this->path, 'size' => $this->size, 'modified' => $this->modified];
    }
}
