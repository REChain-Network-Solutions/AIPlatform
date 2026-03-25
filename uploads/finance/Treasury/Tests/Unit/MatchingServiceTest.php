<?php
namespace Modules\Treasury\Tests\Unit;
use PHPUnit\Framework\TestCase;
use Modules\Treasury\Services\MatchingService;

class MatchingServiceTest extends TestCase
{
    public function testSuggestsHighScoreOnSimilarData(): void
    {
        $svc = new MatchingService();
        $book = json_encode(['amount'=>100.00,'date'=>'2025-09-30','description'=>'ABC Pty Invoice 123']);
        $bank = json_encode(['amount'=>100.00,'date'=>'2025-10-01','description'=>'ABC PTY INV 123']);
        $r = $svc->suggest($book,$bank);
        $this->assertGreaterThanOrEqual(75, $r['score']);
        $this->assertTrue($r['match']);
    }
}
