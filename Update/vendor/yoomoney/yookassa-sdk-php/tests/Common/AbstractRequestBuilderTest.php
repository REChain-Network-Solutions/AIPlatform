<?php

namespace Tests\YooKassa\Common;

use Exception;
use InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use YooKassa\Common\AbstractRequest;
use YooKassa\Common\AbstractRequestBuilder;
use YooKassa\Common\Exceptions\InvalidRequestException;

/**
 * @internal
 */
class AbstractRequestBuilderTest extends TestCase
{
    public function testBuild(): void
    {
        $builder = new TestAbstractRequestBuilder();

        try {
            $builder->build([]);
        } catch (InvalidRequestException $e) {
            $request = $builder->build([
                'isValid' => true,
            ]);
            self::assertTrue($request->isValid());

            $mess = 'test message';

            try {
                $builder->build([
                    'throwException' => new Exception($mess),
                ]);
            } catch (Exception $e) {
                self::assertEquals($mess, $e->getPrevious()->getMessage());

                return;
            }
            self::fail('Exception not thrown in setThrowException method');

            return;
        }
        self::fail('Exception not thrown in build method');
    }

    public function testSetOptions(): void
    {
        $builder = new TestAbstractRequestBuilder();

        $builder->setOptions([]);

        try {
            $builder->build();
        } catch (InvalidRequestException $e) {
            $builder->setOptions([
                'is_valid' => true,
            ]);
            self::assertTrue($builder->build()->isValid());

            try {
                $builder->setOptions('test');
            } catch (Exception $e) {
                self::assertInstanceOf(InvalidArgumentException::class, $e);

                return;
            }
            self::fail('Exception not thrown in setOptions method');
        }
        self::fail('Exception not thrown in build method');
    }
}

class TestAbstractRequestBuilder extends AbstractRequestBuilder
{
    public function setIsValid($value)
    {
        $this->currentObject->setIsValid($value);

        return $this;
    }

    /**
     * @throws Exception
     */
    public function setThrowException(Exception $e): TestAbstractRequestBuilder
    {
        $this->currentObject->setThrowException($e);

        return $this;
    }

    /**
     * Инициализирует пустой запрос
     *
     * @return TestAbstractRequestClass Инстанс запроса, который будем собирать
     */
    protected function initCurrentObject(): TestAbstractRequestClass
    {
        return new TestAbstractRequestClass();
    }
}

class TestAbstractRequestClass extends AbstractRequest
{
    private bool $valid = false;

    private ?Exception $exception = null;

    /**
     * @throws Exception
     */
    public function setThrowException(Exception $e): void
    {
        $this->exception = $e;
    }

    public function setIsValid(bool $value): void
    {
        $this->valid = $value;
    }

    public function isValid(): bool
    {
        return $this->valid;
    }

    /**
     * Валидирует текущий запрос, проверяет все ли нужные свойства установлены.
     *
     * @return bool True если запрос валиден, false если нет
     *
     * @throws Exception
     */
    public function validate(): bool
    {
        if (null !== $this->exception) {
            $this->setValidationError($this->exception->getMessage());

            throw $this->exception;
        }

        return $this->valid;
    }
}
