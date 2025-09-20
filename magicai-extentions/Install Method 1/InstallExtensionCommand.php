<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Services\Extension\Traits\InstallExtension;
use App\Domains\Marketplace\Repositories\ExtensionRepository;

class InstallExtensionCommand extends Command
{
    use InstallExtension;

    protected $signature = 'extension:install {slug}';
    protected $description = 'Install an extension';

    protected ExtensionRepository $extensionRepository; // Declare the property with type hint

    public function __construct(ExtensionRepository $extensionRepository)
    {
        parent::__construct();
        $this->extensionRepository = $extensionRepository; // Inject the repository
    }

    public function handle()
    {
        $slug = $this->argument('slug');

        try {
            $result = $this->install($slug);

            if ($result['status']) {
                $this->info($result['message']);
            } else {
                $this->error($result['message']);
            }
        } catch (\Exception $e) {
            $this->error('Installation failed: ' . $e->getMessage());
        }
    }
}