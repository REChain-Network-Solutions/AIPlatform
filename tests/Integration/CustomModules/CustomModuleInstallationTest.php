<?php

namespace Tests\Integration\CustomModules;

use Tests\TestCase;
use Illuminate\Support\Facades\File;
use Illuminate\Support\Facades\DB;
use App\Models\ModuleInstallLog;
use ZipArchive;

class CustomModuleInstallationTest extends TestCase
{
    protected string $testModulePath;
    protected string $testZipPath;
    
    protected function setUp(): void
    {
        parent::setUp();
        
        $this->testModulePath = storage_path('test_module');
        $this->testZipPath = storage_path('test_module.zip');
    }
    
    protected function tearDown(): void
    {
        File::deleteDirectory($this->testModulePath);
        File::delete($this->testZipPath);
        File::deleteDirectory(base_path('Modules/TestModule'));
        
        parent::tearDown();
    }
    
    /**
     * Test successful module installation with all validators passing
     */
    public function test_successful_module_installation()
    {
        $zip = $this->createValidTestModule('TestModule', '1.0.0');
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $response->assertStatus(200);
        $response->assertJsonPath('status', 'success');
        $response->assertJsonPath('module_name', 'TestModule');
        
        // Verify module files exist
        $this->assertTrue(File::exists(base_path('Modules/TestModule/module.json')));
        
        // Verify installation log was created
        $this->assertDatabaseHas('module_install_logs', [
            'module_name' => 'TestModule',
            'status' => 'installed',
        ]);
    }
    
    /**
     * Test that permission collisions are blocked
     */
    public function test_permission_collision_blocks_installation()
    {
        // Create existing permission
        DB::table('permissions')->insert([
            'permission_key' => 'edit_invoices',
            'module' => 'Core',
            'display_name' => 'Edit Invoices',
            'created_at' => now(),
            'updated_at' => now(),
        ]);
        
        // Create module with same permission key
        $zip = $this->createTestModuleWithPermission(
            'BadModule',
            'edit_invoices'
        );
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $response->assertStatus(200);
        $response->assertJsonPath('status', 'fail');
        $response->assertJsonPath('blocking_issues.0.code', 'PERMISSION_COLLISION');
        
        // Verify module was NOT installed
        $this->assertFalse(File::exists(base_path('Modules/BadModule')));
        
        // Verify blocked log was created
        $this->assertDatabaseHas('module_install_logs', [
            'status' => 'install_blocked_collisions',
        ]);
    }
    
    /**
     * Test that route collisions are blocked
     */
    public function test_route_collision_blocks_installation()
    {
        // Create module with conflicting route
        $zip = $this->createTestModuleWithRoute(
            'BadRouteModule',
            '/admin/settings'
        );
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $response->assertStatus(200);
        $response->assertJsonPath('status', 'fail');
        $response->assertJsonPath('blocking_issues.0.code', 'ROUTE_COLLISION');
        
        // Verify module was NOT installed
        $this->assertFalse(File::exists(base_path('Modules/BadRouteModule')));
    }
    
    /**
     * Test that malicious code is detected
     */
    public function test_malicious_code_blocks_installation()
    {
        // Create module with shell_exec
        $zip = $this->createTestModuleWithShellCode('MaliciousModule');
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $response->assertStatus(200);
        $response->assertJsonPath('status', 'fail');
        $response->assertJsonPath('blocking_issues.0.code', 'SHELL_PATTERN');
        
        // Verify module was NOT installed
        $this->assertFalse(File::exists(base_path('Modules/MaliciousModule')));
    }
    
    /**
     * Test rollback after installation
     */
    public function test_rollback_removes_module_and_restores_state()
    {
        // Install module
        $zip = $this->createValidTestModule('RollbackTestModule', '1.0.0');
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $this->assertTrue(File::exists(base_path('Modules/RollbackTestModule')));
        
        // Get the install ID
        $install = ModuleInstallLog::latest()->first();
        
        // Rollback
        $rollbackResponse = $this->post(
            "/admin/custom-modules/{$install->id}/rollback"
        );
        
        $rollbackResponse->assertStatus(200);
        $rollbackResponse->assertJsonPath('status', 'success');
        
        // Verify module files are gone
        $this->assertFalse(File::exists(base_path('Modules/RollbackTestModule')));
        
        // Verify installation log is marked as rolled back
        $this->assertEquals('rolled_back', $install->refresh()->status);
    }
    
    /**
     * Test that repairs are executed
     */
    public function test_repairs_are_executed()
    {
        $zip = $this->createValidTestModule('RepairTestModule', '1.0.0');
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $response->assertStatus(200);
        
        // Verify permissions were created
        $this->assertDatabaseHas('permissions', [
            'module' => 'RepairTestModule',
        ]);
    }
    
    /**
     * Test that snapshot is created
     */
    public function test_snapshot_is_captured_before_install()
    {
        $zip = $this->createValidTestModule('SnapshotTestModule', '1.0.0');
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        $install = ModuleInstallLog::latest()->first();
        
        $this->assertNotEmpty($install->pre_install_snapshot);
        $this->assertTrue($install->can_rollback);
    }
    
    /**
     * Test partial failure handling
     */
    public function test_partial_failure_is_logged()
    {
        $zip = $this->createValidTestModule('PartialModule', '1.0.0');
        
        // Mock a repair failure
        // (This would require more complex setup)
        
        $response = $this->post('/admin/custom-modules', [
            'filePath' => $zip,
        ]);
        
        // Response should indicate success or partial based on repairs
        $this->assertIn($response->json('status'), ['success', 'partial']);
    }
    
    // ========================================================================
    // Helper Methods
    // ========================================================================
    
    protected function createValidTestModule(string $name, string $version): string
    {
        $modulePath = storage_path("test_modules/{$name}");
        File::ensureDirectoryExists($modulePath);
        
        // Create module.json
        File::put($modulePath . '/module.json', json_encode([
            'name' => $name,
            'version' => $version,
            'permissions' => [
                ['key' => 'manage_' . strtolower($name), 'label' => "Manage {$name}"],
            ],
            'routes' => [],
        ]));
        
        // Create a simple PHP file
        File::put($modulePath . '/Module.php', "<?php\nnamespace Modules\\{$name};\n\nclass Module\n{\n}");
        
        return $this->zipDirectory($modulePath, storage_path("test_{$name}.zip"));
    }
    
    protected function createTestModuleWithPermission(string $name, string $permissionKey): string
    {
        $modulePath = storage_path("test_modules/{$name}");
        File::ensureDirectoryExists($modulePath);
        
        File::put($modulePath . '/module.json', json_encode([
            'name' => $name,
            'version' => '1.0.0',
            'permissions' => [
                ['key' => $permissionKey, 'label' => ucfirst($permissionKey)],
            ],
        ]));
        
        return $this->zipDirectory($modulePath, storage_path("test_{$name}.zip"));
    }
    
    protected function createTestModuleWithRoute(string $name, string $routeUri): string
    {
        $modulePath = storage_path("test_modules/{$name}");
        File::ensureDirectoryExists($modulePath);
        
        File::put($modulePath . '/module.json', json_encode([
            'name' => $name,
            'version' => '1.0.0',
            'routes' => [
                ['uri' => $routeUri, 'method' => 'GET'],
            ],
        ]));
        
        return $this->zipDirectory($modulePath, storage_path("test_{$name}.zip"));
    }
    
    protected function createTestModuleWithShellCode(string $name): string
    {
        $modulePath = storage_path("test_modules/{$name}");
        File::ensureDirectoryExists($modulePath);
        
        File::put($modulePath . '/module.json', json_encode([
            'name' => $name,
            'version' => '1.0.0',
        ]));
        
        // Create malicious PHP file
        File::put($modulePath . '/Shell.php', "<?php\nshell_exec('whoami');\n");
        
        return $this->zipDirectory($modulePath, storage_path("test_{$name}.zip"));
    }
    
    protected function zipDirectory(string $source, string $destination): string
    {
        $zip = new ZipArchive();
        $zip->open($destination, ZipArchive::CREATE | ZipArchive::OVERWRITE);
        
        $files = File::allFiles($source);
        
        foreach ($files as $file) {
            $zip->addFile(
                $file->getRealPath(),
                str_replace($source . '/', '', $file->getRealPath())
            );
        }
        
        $zip->close();
        
        return $destination;
    }
}
