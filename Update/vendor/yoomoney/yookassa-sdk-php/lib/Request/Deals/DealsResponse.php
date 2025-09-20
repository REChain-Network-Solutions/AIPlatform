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

namespace YooKassa\Request\Deals;

use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Model\Deal\SafeDeal;
use YooKassa\Request\AbstractListResponse;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель DealsResponse.
 *
 * Класс объекта ответа от API со списком сделок магазина.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property SafeDeal[]|ListObjectInterface|null $items Список сделок, созданных за последние три года. Сделки отсортированы по времени создания в порядке убывания (от новых к старым).
 */
class DealsResponse extends AbstractListResponse
{
    /**
     * Список сделок, созданных за последние три года.
     * Сделки отсортированы по времени создания в порядке убывания (от новых к старым).
     * Если результатов больше, чем задано в limit, список будет выводиться фрагментами.
     * В этом случае в ответе на запрос вернется фрагмент списка и параметр next_cursor с указателем на следующий фрагмент.
     *
     * @var SafeDeal[]|ListObjectInterface|null Список сделок
     */
    #[Assert\Valid]
    #[Assert\AllType(SafeDeal::class)]
    #[Assert\Type(ListObject::class)]
    protected ?ListObject $_items = null;

    /**
     * Возвращает массив сделок.
     *
     * @return SafeDeal[]|ListObjectInterface Список сделок
     */
    public function getItems(): ListObjectInterface
    {
        if($this->_items === null) {
            $this->_items = new ListObject(SafeDeal::class);
        }
        return $this->_items;
    }
}
