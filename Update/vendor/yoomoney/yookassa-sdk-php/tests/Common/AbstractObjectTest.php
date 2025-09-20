<?php

namespace Tests\YooKassa\Common;

use DateTime;
use PHPUnit\Framework\TestCase;
use stdClass;
use YooKassa\Common\AbstractObject;

/**
 * @internal
 */
class AbstractObjectTest extends TestCase
{
    /**
     * @dataProvider offsetDataProvider
     *
     * @param mixed $value
     * @param mixed $exists
     */
    public function testOffsetExists(mixed $value, mixed $exists): void
    {
        $instance = $this->getTestInstance();
        if (!$exists) {
            self::assertFalse($instance->offsetExists($value));
            self::assertFalse(isset($instance[$value]));
            self::assertFalse(isset($instance->{$value}));

            $instance->offsetSet($value, $value);
        }
        self::assertTrue($instance->offsetExists($value));
        self::assertTrue(isset($instance[$value]));
        self::assertTrue(isset($instance->{$value}));
    }

    /**
     * @dataProvider offsetDataProvider
     *
     * @param mixed $value
     */
    public function testOffsetGet(mixed $value): void
    {
        $instance = $this->getTestInstance();

        $tmp = $instance->offsetGet($value);
        self::assertNull($tmp);
        $tmp = $instance[$value];
        self::assertNull($tmp);
        $tmp = $instance->{$value};
        self::assertNull($tmp);

        $instance->offsetSet($value, $value);

        $tmp = $instance->offsetGet($value);
        self::assertEquals($value, $tmp);
        $tmp = $instance[$value];
        self::assertEquals($value, $tmp);
        $tmp = $instance->{$value};
        self::assertEquals($value, $tmp);
    }

    /**
     * @dataProvider offsetDataProvider
     *
     * @param mixed $value
     * @param mixed $exists
     */
    public function testOffsetUnset(mixed $value, mixed $exists): void
    {
        $instance = $this->getTestInstance();
        if ($exists) {
            self::assertTrue($instance->offsetExists($value));
            $instance->offsetUnset($value);
            self::assertTrue($instance->offsetExists($value));
            unset($instance[$value]);
            self::assertTrue($instance->offsetExists($value));
            unset($instance->{$value});
            self::assertTrue($instance->offsetExists($value));
        } else {
            self::assertFalse($instance->offsetExists($value));
            $instance->offsetUnset($value);
            self::assertFalse($instance->offsetExists($value));
            unset($instance[$value]);
            self::assertFalse($instance->offsetExists($value));
            unset($instance->{$value});
            self::assertFalse($instance->offsetExists($value));

            $instance->{$value} = $value;
            self::assertTrue($instance->offsetExists($value));
            $instance->offsetUnset($value);
            self::assertFalse($instance->offsetExists($value));

            $instance->{$value} = $value;
            self::assertTrue($instance->offsetExists($value));
            unset($instance[$value]);
            self::assertFalse($instance->offsetExists($value));

            $instance->{$value} = $value;
            self::assertTrue($instance->offsetExists($value));
            unset($instance->{$value});
            self::assertFalse($instance->offsetExists($value));
        }
    }

    public static function offsetDataProvider(): array
    {
        return [
            ['property', true],
            ['propertyCamelCase', true],
            ['property_camel_case', true],
            ['Property', true],
            ['PropertyCamelCase', true],
            ['Property_Camel_Case', true],
            ['not_exists', false],
        ];
    }

    public function testJsonSerialize(): void
    {
        $instance = $this->getTestInstance();

        $data = $instance->jsonSerialize();
        $expected = [];
        self::assertEquals($expected, $data);

        $instance->setProperty('propertyValue');
        $data = $instance->jsonSerialize();
        $expected = [
            'property' => 'propertyValue',
        ];
        self::assertEquals($expected, $data);

        $instance->setPropertyCamelCase($this->getTestInstance());
        $data = $instance->jsonSerialize();
        $expected = [
            'property' => 'propertyValue',
            'property_camel_case' => [],
        ];
        self::assertEquals($expected, $data);

        $instance->getPropertyCamelCase()->setProperty(['test', 1, 2, 3]);
        $data = $instance->jsonSerialize();
        $expected = [
            'property' => 'propertyValue',
            'property_camel_case' => [
                'property' => ['test', 1, 2, 3],
            ],
        ];
        self::assertEquals($expected, $data);

        $date = new DateTime();
        $instance->getPropertyCamelCase()->setPropertyCamelCase($date);
        $data = $instance->jsonSerialize();
        $expected = [
            'property' => 'propertyValue',
            'property_camel_case' => [
                'property' => ['test', 1, 2, 3],
                'property_camel_case' => $date->format(YOOKASSA_DATE),
            ],
        ];
        self::assertEquals($expected, $data);

        $instance->getPropertyCamelCase()->unknown_obj = $this->getTestInstance();
        $data = $instance->jsonSerialize();
        $expected = [
            'property' => 'propertyValue',
            'property_camel_case' => [
                'property' => ['test', 1, 2, 3],
                'property_camel_case' => $date->format(YOOKASSA_DATE),
                'unknown_obj' => [],
            ],
        ];
        self::assertEquals($expected, $data);

        $instance->unknown_property = true;
        $data = $instance->jsonSerialize();
        $expected['unknown_property'] = true;
        self::assertEquals($expected, $data);

        $instance->unknown_array = [false];
        $data = $instance->jsonSerialize();
        $expected['unknown_array'] = [false];
        self::assertEquals($expected, $data);

        $instance->unknown_date = $date;
        $data = $instance->jsonSerialize();
        $expected['unknown_date'] = $date->format(YOOKASSA_DATE);
        self::assertEquals($expected, $data);

        $obj = new stdClass();
        $obj->test = 'test1';
        $instance->unknown_obj = $obj;
        $data = $instance->jsonSerialize();
        $expected['unknown_obj'] = $obj;
        self::assertEquals($expected, $data);

        $obj = new stdClass();
        $obj->test = 'test2';
        $instance->property_camel_case = $obj;
        $data = $instance->jsonSerialize();
        $expected['property_camel_case'] = $obj;
        self::assertEquals($expected, $data);

        $instance->property_camel_case = null;
        $data = $instance->jsonSerialize();
        unset($expected['property_camel_case']);
        self::assertEquals($expected, $data);
    }

    protected function getTestInstance(): TestAbstractObject
    {
        return new TestAbstractObject();
    }
}

class TestAbstractObject extends AbstractObject
{
    private $_property;
    private $_anotherProperty;

    public function getProperty()
    {
        return $this->_property;
    }

    public function setProperty($value): void
    {
        $this->_property = $value;
    }

    public function getPropertyCamelCase()
    {
        return $this->_anotherProperty;
    }

    public function setPropertyCamelCase($value): void
    {
        $this->_anotherProperty = $value;
    }
}
