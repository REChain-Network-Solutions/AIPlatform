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

namespace YooKassa\Request\Receipts;

use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Request\AbstractListResponse;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс для работы со списком чеков.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property AbstractReceiptResponse[]|ListObjectInterface|null $items Список чеков, созданных за последние три года. Чеки отсортированы по времени создания в порядке убывания (от новых к старым).
 */
class ReceiptsResponse extends AbstractListResponse
{
    /**
     * Список чеков, созданных за последние три года.
     * Чеки отсортированы по времени создания в порядке убывания (от новых к старым).
     * Если результатов больше, чем задано в limit, список будет выводиться фрагментами.
     * В этом случае в ответе на запрос вернется фрагмент списка и параметр next_cursor с указателем на следующий фрагмент.
     *
     * @var AbstractReceiptResponse[]|ListObjectInterface|null Список чеков
     */
    #[Assert\Valid]
    #[Assert\Type(ListObject::class)]
    #[Assert\AllType(AbstractReceiptResponse::class)]
    protected ?ListObject $_items = null;

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
            $factory = new ReceiptResponseFactory();

            $itemsObj = [];
            foreach ($sourceArray['items'] as $item) {
                if ($receipt = $factory->factory($item)) {
                    $itemsObj[] = $receipt;
                }
            }

            $this->_items = $this->validatePropertyValue('_items', $itemsObj);
        }
        if (!empty($sourceArray['next_cursor'])) {
            $this->_next_cursor = $this->validatePropertyValue('_next_cursor', $sourceArray['next_cursor']);
        }
    }

    /**
     * Возвращает список чеков.
     *
     * @return AbstractReceiptResponse[]|ListObjectInterface Список чеков
     */
    public function getItems(): ListObjectInterface
    {
        if($this->_items === null) {
            $this->_items = new ListObject(AbstractReceiptResponse::class);
        }
        return $this->_items;
    }
}
