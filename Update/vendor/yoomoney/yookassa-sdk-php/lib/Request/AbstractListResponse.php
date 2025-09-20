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

namespace YooKassa\Request;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Common\ListObjectInterface;

/**
 * Абстрактный класс для объектов, содержащих список объектов-моделей в ответе на запрос.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Формат выдачи результатов запроса. Возможное значение: `list` (список).
 * @property string $nextCursor Указатель на следующий фрагмент списка. Обязательный параметр, если размер списка больше размера выдачи (`limit`) и конец выдачи не достигнут.
 */
abstract class AbstractListResponse extends AbstractObject
{
    /**
     * Формат выдачи результатов запроса. Возможное значение: `list` (список).
     *
     * @var string
     */
    #[Assert\Type('string')]
    #[Assert\Choice(['list'])]
    protected string $_type = 'list';

    /**
     * Указатель на следующий фрагмент списка. Обязательный параметр, если размер списка больше размера выдачи (`limit`) и конец выдачи не достигнут.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    protected ?string $_next_cursor = null;

    /**
     * @param iterable $sourceArray
     * @return void
     */
    public function fromArray(iterable $sourceArray): void
    {
        if (!empty($sourceArray['type'])) {
            $this->_type = $this->validatePropertyValue('_type', $sourceArray['type']);
        }
        if (!empty($sourceArray['items'])) {
            $this->_items = $this->validatePropertyValue('_items', $sourceArray['items']);
        }
        if (!empty($sourceArray['next_cursor'])) {
            $this->_next_cursor = $this->validatePropertyValue('_next_cursor', $sourceArray['next_cursor']);
        }
    }

    /**
     * Возвращает формат выдачи результатов запроса.
     *
     * @return string|null
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Возвращает токен следующей страницы, если он задан, или null.
     *
     * @return null|string Токен следующей страницы
     */
    public function getNextCursor(): ?string
    {
        return $this->_next_cursor;
    }

    /**
     * Проверяет, имеется ли в ответе токен следующей страницы.
     *
     * @return bool True если токен следующей страницы есть, false если нет
     */
    public function hasNextCursor(): bool
    {
        return null !== $this->_next_cursor;
    }

    /**
     * Возвращает список объектов в ответе на запрос
     *
     * @return ListObjectInterface
     */
    abstract public function getItems(): ListObjectInterface;
}
