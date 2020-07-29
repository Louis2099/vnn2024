mutable struct Tree{T}
    data::Vector{T}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    leaves::Set{Int}
end
Tree(data) = Tree{typeof(data)}([data], [0], [Vector{Int}()], Set([1]))

function add_child!(t::Tree, parent::Int, data)
    push!(t.data, data)
    push!(t.children, Vector{Int}())
    x = length(t.data)
    push!(t.leaves, x)
    push!(t.parent, parent)
    push!(t.children[parent], x)
    in(parent, t.leaves) && pop!(t.leaves, parent)
    return x
end

function print_tree(t::Tree, x::Int = 1)
    for c in t.children[x]
        println(t.data[x], "->", t.data[c])
    end
    for c in t.children[x]
        print_tree(t, c)
    end
end

function is_leaf(t::Tree, x::Int)
    return in(x, t.leaves)
end

function size(t::Tree)
    return length(t.data)
end