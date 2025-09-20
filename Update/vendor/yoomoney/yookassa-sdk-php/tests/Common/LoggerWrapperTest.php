<?php

namespace Tests\YooKassa\Common;

use PHPUnit\Framework\TestCase;
use Psr\Log\InvalidArgumentException;
use Psr\Log\LoggerInterface;
use Psr\Log\LogLevel;
use stdClass;
use TypeError;
use YooKassa\Common\LoggerWrapper;
use YooKassa\Helpers\Random;

/**
 * @internal
 */
class LoggerWrapperTest extends TestCase
{
    public function testConstruct(): void
    {
        $logger = new LoggerWrapper(new ArrayLogger());
        self::assertNotNull($logger);
        $logger = new LoggerWrapper(static function ($level, $message, $context): void {
        });
        self::assertNotNull($logger);
    }

    /**
     * @dataProvider invalidLoggerDataProvider
     */
    public function testInvalidConstruct(mixed $source): void
    {
        $this->expectException(TypeError::class);
        $this->expectException(InvalidArgumentException::class);
        new LoggerWrapper($source);
    }

    public static function invalidLoggerDataProvider(): array
    {
        return [
            [new stdClass()],
            [true],
            [false],
            [[]],
            [1],
            ['test'],
        ];
    }

    /**
     * @dataProvider validLoggerDataProvider
     */
    public function testLog(string $level, string $message, array $context): void
    {
        $wrapped = new ArrayLogger();

        $instance = new LoggerWrapper($wrapped);
        $instance->log($level, $message, $context);
        $expected = [$level, $message, $context];
        self::assertEquals($expected, $wrapped->getLastLog());

        $wrapped = new ArrayLogger();
        $instance = new LoggerWrapper(function ($level, $message, $context) use ($wrapped): void {
            $wrapped->log($level, $message, $context);
        });
        $instance->log($level, $message, $context);
        $expected = [$level, $message, $context];
        self::assertEquals($expected, $wrapped->getLastLog());
    }

    /**
     * @dataProvider validLoggerDataProvider
     */
    public function testLogMethods(string $notUsed, string $message, array $context): void
    {
        $methodsMap = [
            LogLevel::EMERGENCY => 'emergency',
            LogLevel::ALERT => 'alert',
            LogLevel::CRITICAL => 'critical',
            LogLevel::ERROR => 'error',
            LogLevel::WARNING => 'warning',
            LogLevel::NOTICE => 'notice',
            LogLevel::INFO => 'info',
            LogLevel::DEBUG => 'debug',
        ];

        $wrapped = new ArrayLogger();
        $instance = new LoggerWrapper($wrapped);
        foreach ($methodsMap as $level => $method) {
            $instance->{$method}($message, $context);
            $expected = [$level, $message, $context];
            self::assertEquals($expected, $wrapped->getLastLog());
        }
    }

    public static function validLoggerDataProvider(): array
    {
        return [
            [LogLevel::EMERGENCY, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::ALERT, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::CRITICAL, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::ERROR, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::WARNING, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::NOTICE, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::INFO, Random::str(10, 20), [Random::str(10, 20)]],
            [LogLevel::DEBUG, Random::str(10, 20), [Random::str(10, 20)]],
        ];
    }
}

class ArrayLogger implements LoggerInterface
{
    private array $lastLog;

    public function log($level, $message, array $context = []): void
    {
        $this->lastLog = [$level, $message, $context];
    }

    public function getLastLog(): array
    {
        return $this->lastLog;
    }

    public function emergency($message, array $context = []): void
    {
        $this->log(LogLevel::EMERGENCY, $message, $context);
    }

    public function alert($message, array $context = []): void
    {
        $this->log(LogLevel::ALERT, $message, $context);
    }

    public function critical($message, array $context = []): void
    {
        $this->log(LogLevel::CRITICAL, $message, $context);
    }

    public function error($message, array $context = []): void
    {
        $this->log(LogLevel::ERROR, $message, $context);
    }

    public function warning($message, array $context = []): void
    {
        $this->log(LogLevel::WARNING, $message, $context);
    }

    public function notice($message, array $context = []): void
    {
        $this->log(LogLevel::NOTICE, $message, $context);
    }

    public function info($message, array $context = []): void
    {
        $this->log(LogLevel::INFO, $message, $context);
    }

    public function debug($message, array $context = []): void
    {
        $this->log(LogLevel::DEBUG, $message, $context);
    }
}
