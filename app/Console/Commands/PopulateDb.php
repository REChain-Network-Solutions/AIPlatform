<?php

namespace App\Console\Commands;

use Exception;
use Illuminate\Console\Command;
use RachidLaasri\LaravelInstaller\Helpers\DatabaseManager;

class PopulateDb extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'app:populate-db';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Run database migrations and seed the database using DatabaseManager';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $this->info('Starting database migration and seeding...');

        try {
            $databaseManager = new DatabaseManager;
            $response = $databaseManager->migrateAndSeed();

            if ($response) {
                $this->info('Database migrated and seeded successfully.');
            } else {
                $this->warn('Migration and seeding completed with warnings or messages:');
                $this->line($response);
            }

        } catch (Exception $e) {
            $this->error('An error occurred while migrating and seeding the database:');
            $this->error($e->getMessage());

            return 1;
        }

        return 0;
    }
}
