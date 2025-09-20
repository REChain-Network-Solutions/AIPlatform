<?php

namespace Tests\YooKassa\Model\Payment\Confirmation;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Model\Payment\Confirmation\ConfirmationCodeVerification;

/**
 * @internal
 */
class ConfirmationCodeVerificationTest extends AbstractTestCase
{
    protected ConfirmationCodeVerification $object;

    /**
     * @return ConfirmationCodeVerification
     */
    protected function getTestInstance(): ConfirmationCodeVerification
    {
        return new ConfirmationCodeVerification();
    }

    /**
     * @return void
     */
    public function testConfirmationCodeVerificationClassExists(): void
    {
        $this->object = $this->getMockBuilder(ConfirmationCodeVerification::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(ConfirmationCodeVerification::class));
        $this->assertInstanceOf(ConfirmationCodeVerification::class, $this->object);
    }

    /**
     * Test property "type"
     *
     * @return void
     * @throws Exception
     */
    public function testType(): void
    {
        $instance = $this->getTestInstance();
        self::assertContains($instance->getType(), ['code_verification']);
        self::assertContains($instance->type, ['code_verification']);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
    }

    /**
     * Test invalid property "type"
     * @dataProvider invalidTypeDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }
}
