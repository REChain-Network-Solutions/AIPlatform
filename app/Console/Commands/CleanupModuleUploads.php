<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\Facades\File;

class CleanupModuleUploads extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'modules:cleanup-uploads {--days=7 : Delete uploads older than this many days}';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Remove old module upload ZIP files to free disk space';

    /**
     * Execute the console command.
     */
    public function handle(): int
    {
        $days = (int) $this->option('days');
        $uploadPath = config('froiden_envato.tmp_path');
        
        if (!File::exists($uploadPath)) {
            $this->info("Upload directory does not exist: {$uploadPath}");
            return 0;
        }
        
        $cutoffTime = now()->subDays($days)->timestamp;
        $deleted = 0;
        $errors = [];
        
        $this->info("Scanning for uploads older than {$days} days...");
        
        try {
            $files = File::files($uploadPath);
            
            if (empty($files)) {
                $this->info("No files found in upload directory.");
                return 0;
            }
            
            $this->withProgressBar($files, function ($file) use (&$deleted, &$errors, $cutoffTime) {
                try {
                    $fileTime = $file->getATime();
                    
                    if ($fileTime < $cutoffTime) {
                        File::delete($file->getPathname());
                        $deleted++;
                    }
                } catch (\Exception $e) {
                    $errors[] = [
                        'file' => $file->getFilename(),
                        'error' => $e->getMessage(),
                    ];
                }
            });
            
            $this->newLine();
            
        } catch (\Exception $e) {
            $this->error("Error scanning upload directory: {$e->getMessage()}");
            return 1;
        }
        
        // Summary
        $this->info("Cleanup complete!");
        $this->line("  • Files deleted: {$deleted}");
        $this->line("  • Errors: " . count($errors));
        
        if (!empty($errors)) {
            $this->warn("Errors occurred during cleanup:");
            foreach ($errors as $error) {
                $this->line("  - {$error['file']}: {$error['error']}");
            }
        }
        
        \Log::info("Module upload cleanup executed", [
            'days' => $days,
            'deleted' => $deleted,
            'errors' => count($errors),
        ]);
        
        return 0;
    }
}
