from __future__ import absolute_import, division, print_function


def blockify(input_list):

    # initialize return lists
    block_items = []
    n_per_block = []

    # initialize current item and block
    curr_item = None
    curr_block_length = 0

    # loop through list
    for i, item in enumerate(input_list):

        # determine item in first block
        if curr_item is None:
            curr_item = item

        # check if new block starts
        if item != curr_item:
            block_items.append(curr_item)
            n_per_block.append(curr_block_length)
            curr_item = item
            curr_block_length = 0

        # increment current block length
        curr_block_length += 1

    # add item and length of last block
    block_items.append(curr_item)
    n_per_block.append(curr_block_length)

    # return lists
    return block_items, n_per_block
