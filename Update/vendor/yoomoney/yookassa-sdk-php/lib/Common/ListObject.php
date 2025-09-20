<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Common;

use Ds\Vector;
use InvalidArgumentException;
use ReflectionClass;

/**
 * Класс, представляющий модель ListObject.
 *
 * Коллекция объектов AbstractObject.
 *
 * @example 05-collections.php 3 Работа с коллекциями объектов
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class ListObject implements ListObjectInterface
{
    /**
     * @var string|null Тип хранимых объектов
     */
    private ?string $type = null;

    /**
     * @var Vector Коллекция объектов
     */
    private Vector $list;

    /**
     * @var bool Флаг, является ли тип хранимых объектов инстанцируемым.
     * Если тип не инстанцируемый, в коллекцию не удастся добавить массив для создания объекта коллекции
     */
    private bool $isTypeInstantiable = false;

    /**
     * @param string $type Тип хранимых объектов
     * @param array|null $data Массив данных
     */
    public function __construct(string $type, ?array $data = [])
    {
        $this->list = new Vector();

        $this->setType($type);

        if (!empty($data)) {
            foreach ($data as $item) {
                $this->add($item);
            }
        }
    }

    /**
     * Возвращает тип объектов в коллекции
     *
     * @return string
     */
    public function getType(): string
    {
        return $this->type;
    }

    /**
     * Устанавливает тип объектов в коллекции
     *
     * @param string $type
     * @return $this
     */
    public function setType(string $type): self
    {
        if ($this->type !== null && $type !== $this->type && !$this->isEmpty()) {
            throw new InvalidArgumentException('You cannot change the type of a non-empty ListObject');
        }
        if (class_exists($type) && is_subclass_of($type, AbstractObject::class)) {
            $this->type = $type;
            $this->isTypeInstantiable = (new ReflectionClass($type))->isInstantiable();
        } else {
            throw new InvalidArgumentException('Invalid item type for ListObject');
        }

        return $this;
    }

    /**
     * Добавляет объект в коллекцию
     *
     * @param mixed $item
     * @return $this
     */
    public function add(mixed $item): self
    {
        if (is_array($item)) {
            if (!$this->isTypeInstantiable) {
                throw new InvalidArgumentException(
                    'You cannot add item-array to a ListObject if the type is not instantiable'
                );
            }
            $this->list->push(new $this->type($item));
            return $this;
        }

        if ($item instanceof $this->type) {
            $this->list->push($item);
        } else {
            throw new InvalidArgumentException(
                'You cannot add not ' . $this->type . ' object to a ListObject'
            );
        }

        return $this;
    }

    /**
     * Добавляет массив объектов в коллекцию
     *
     * @param iterable $data
     * @return $this
     */
    public function merge(iterable $data): self
    {
        foreach ($data as $item) {
            $this->add($item);
        }

        return $this;
    }

    /**
     * Удаляет объект из коллекции по индексу
     *
     * @param int $index
     * @return $this
     */
    public function remove(int $index): self
    {
        $this->list->offsetUnset($index);

        return $this;
    }

    /**
     * Очищает коллекцию
     *
     * @return $this
     */
    public function clear(): self
    {
        $this->list->clear();

        return $this;
    }

    /**
     * Проверка на пустую коллекцию
     *
     * @return bool
     */
    public function isEmpty(): bool
    {
        return $this->list->isEmpty();
    }

    /**
     * Возвращает коллекцию
     *
     * @return Vector
     */
    public function getItems(): Vector
    {
        return $this->list;
    }

    /**
     * Возвращает объект коллекции по индексу
     *
     * @param int $index
     * @return AbstractObject
     */
    public function get(int $index): AbstractObject
    {
        return $this->list->get($index);
    }

    /**
     * Возвращает количество объектов в коллекции
     *
     * @return int
     */
    public function count(): int
    {
        return $this->list->count();
    }

    #[\ReturnTypeWillChange]
    public function offsetExists(mixed $offset)
    {
        return $this->list->offsetExists($offset);
    }

    #[\ReturnTypeWillChange]
    public function offsetGet(mixed $offset)
    {
        return $this->list->offsetGet($offset);
    }

    #[\ReturnTypeWillChange]
    public function offsetSet(mixed $offset, mixed $value)
    {
        $this->list->offsetSet($offset, $value);
    }

    #[\ReturnTypeWillChange]
    public function offsetUnset(mixed $offset)
    {
        $this->list->offsetUnset($offset);
    }

    /**
     * Возвращает коллекцию в виде массива
     *
     * @return array
     */
    public function jsonSerialize(): array
    {
        $result = [];
        foreach ($this->list as $item) {
            $result[] = $item->jsonSerialize();
        }
        return $result;
    }

    /**
     * Возвращает коллекцию в виде массива
     *
     * @return array
     */
    public function toArray(): array
    {
        return $this->jsonSerialize();
    }

    /**
     * @return Vector
     */
    public function getIterator(): Vector
    {
        return $this->list;
    }
}
